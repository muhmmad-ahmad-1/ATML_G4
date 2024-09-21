import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from transformers import DistilBertTokenizer
from models.transforms import transforms_clip_vit

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def move_tensors_to_device(d: dict, device: torch.device) -> dict:
    """
    Helper function for moving tensors inside (nested) dictionaries to a target device - not necessary to use but can be useful depending on implementation
    
    :param d: Dictionary with potential nested dictionaries and tensors.
    :param device: The device to move tensors to (e.g., torch.device('cuda:0') or torch.device('cpu')).
    :return: A new dictionary with tensors moved to the specified device.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively process nested dictionaries
            new_dict[k] = move_tensors_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            # Move tensors to the device
            new_dict[k] = v.to(device)
        else:
            # For non-tensor, non-dict items, just copy them as is
            new_dict[k] = v
    return new_dict

def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str,
               modeltype: str):
    model = model.to(device)
    model.train()
    total_loss = 0
    pred_list = None

    for i,batch in enumerate(dataloader):
        if modeltype != "clip":
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred,y_batch.reshape(-1).long())
            total_loss += loss.item()
            pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch
        
        else:
            optimizer.zero_grad()
            batch = move_tensors_to_device(batch,device)
            preds = model(batch)
            labels = torch.arange(preds.shape[0]).to(device)
            loss_i = loss(preds, labels) 
            loss_t = loss(preds.T, labels) 
            loss = (loss_i + loss_t)/2
            total_loss += loss.item()

        if i % (len(dataloader)//20) == 0 or i == len(dataloader)-1:
            print("Batch",i,":")
            print(f"Loss: {total_loss/(i+1):.4f}")
            accuracy = pred_list.float().mean().cpu().numpy()
            print(f"Accuracy: {accuracy:.4f}")

        loss.backward()
        optimizer.step()

    return total_loss/len(dataloader),pred_list.float().mean().cpu().numpy()


@torch.inference_mode
def eval_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               device: str,
               modeltype: str,
               data:str):
    model = model.to(device)

    model.eval()

    pred_list = None
    total_loss = 0

    if modeltype == 'clip' and data == "cifar10":        
        captions = cifar10_label_to_text([0,1,2,3,4,5,6,7,8,9])
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encoded_captions = tokenizer(captions, padding=True, truncation=True, max_length = 200)
        encoded_captions =  {key: torch.tensor(values) for key, values in encoded_captions.items()}
                
    for i,batch in enumerate(dataloader):
        if modeltype != 'clip':
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            pred = model(x_batch)

            loss = loss_fn(pred,y_batch.reshape(-1).long())
            
        else:
            batch["captions"] = encoded_captions
            batch = move_tensors_to_device(batch,device)
            preds = model(batch)
            labels = torch.arange(preds.shape[0]).to(device)
            loss = loss_fn(preds, labels) 
            total_loss += loss.item()
        
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

        total_loss += loss.item()
            
        if i % (len(dataloader)//20) == 0 or i == len(dataloader)-1:
            print("Batch",i,":")
            print(f"Loss: {total_loss/(i+1):.4f}")
            accuracy = pred_list.float().mean().cpu().numpy()
            print(f"Accuracy: {accuracy:.4f}")

    total_loss /= len(dataloader)

    return total_loss,pred_list.float().mean().cpu().numpy()

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class CLIPDataset(Dataset):
    def __init__(
        self, dataset, images, captions,transform, max_length
    ):
        """
        Initializes the dataset with image filenames, captions, a tokenizer function, and optional image transforms.
        
        :param images: PIL images
        :param captions: List of captions corresponding to the images.
        :param transform: Optional transform to be applied on images.
        :param max_length: maximum length of sentence to be tokenized
        """
        self.images = images
        self.captions = captions
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transform
        self.dataset = dataset

        # Tokenize all captions
        self.encoded_captions = self.tokenizer(captions, padding=True, truncation=True, max_length = max_length)
    
    def __getitem__(self, idx: int):
        """
        Retrieves an item from the dataset given an index.

        :param idx: Index of the item to retrieve.
        :return: Dictionary with 'image' and 'caption' keys.
        """
        # Get encoded caption
        encoded_caption = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
        
        if self.transform:
            image = self.transform(image)
        
        # Return dictionary with tensors
        return {
            'image': image,
            'caption': encoded_caption
        }
    
    def __len__(self):
        """
        Returns the number of items in the dataset.

        :return: Number of items.
        """
        return len(self.captions)

def cifar10_label_to_text(labels):
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return [f"This is a photo of a {class_names[label]}." for label in labels]

class CIFAR10_CLIPDataset(CLIPDataset):
    def __init__(self,dataset, transform, max_length):
        super(CLIPDataset,self).__init__()
        self.captions = cifar10_label_to_text(dataset.targets)
        self.unique_captions = cifar10_label_to_text([0,1,2,3,4,5,6,7,8,9])
        self.dataset = CLIPDataset(dataset.data,self.captions,transform,max_length)
    
    def data(self):
        return self.dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.data = images
        self.targets = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

#`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
class DataLoaders(DataLoader):
    def __init__(
        self,train_dataset,test_dataset,model_type,batch_size,shuffle,data
    ):
        if model_type != "clip":    
            self.train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
            self.test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)
        else:
            if data == "cifar10":
                images = train_dataset.data      
                labels = train_dataset.targets 

                class_images = {i: [] for i in range(10)}

                for img, label in zip(images, labels):
                    class_images[label].append(img)

                shuffled_images = []
                shuffled_labels = []

                for i in range(len(class_images[0])):
                    for class_label in range(10):
                        shuffled_images.append(class_images[class_label][i])
                        shuffled_labels.append(class_label)
                train = CustomDataset(shuffled_images,shuffled_labels)
                self.train_dataset = CIFAR10_CLIPDataset(train,transform=transforms_clip_vit,max_length=200)
                self.test_dataset = CIFAR10_CLIPDataset(test_dataset,transform=transforms_clip_vit,max_length=200)
                self.train_loader = DataLoader(self.train_dataset.dataset,batch_size=10,shuffle=False)
                self.test_loader = DataLoader(self.test_dataset.dataset,batch_size=batch_size,shuffle=True)
    def get_loaders(self):
        return self.train_loader, self.test_loader
    
        


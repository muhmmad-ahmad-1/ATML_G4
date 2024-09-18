import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from transformers import DistilBertTokenizer

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str):
    model = model.to(device)
    model.train()
    total_loss = 0
    pred_list = None

    for i,(x_batch,y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        pred = model(x_batch)
        loss = loss_fn(pred,y_batch.reshape(-1).long())
        total_loss += loss.item()
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

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
               device: str):
    model = model.to(device)

    model.eval()

    pred_list = None
    total_loss = 0

    for i,(x_batch,y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        pred = model(x_batch)

        loss = loss_fn(pred,y_batch.reshape(-1).long())

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
        self, images, captions,transform, max_length
    ):
        """
        Initializes the dataset with image filenames, captions, a tokenizer function, and optional image transforms.
        
        :param images: PIL images
        :param captions: List of captions corresponding to the images.
        :param tokenizer: Function to tokenize captions. It should return a dictionary with tensors.
        :param transform: Optional transform to be applied on images.
        :param max_length: maximum length of sentence to be tokenized
        """
        self.images = images
        self.captions = captions
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.transform = transform

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
    def __init__(self,images, labels, transform, max_length):
        super(CLIPDataset,self).__init__()
        self.captions = cifar10_label_to_text(labels)
        self.dataset = CLIPDataset(images,self.captions,transform,max_length)
    
    def data(self):
        return self.dataset
    
    
        


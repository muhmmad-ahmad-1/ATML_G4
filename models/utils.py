import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from transformers import DistilBertTokenizer
from models.transforms import transforms_clip_vit, transforms_resnet
from torchvision import datasets
import os
import random

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
               modeltype: str,
               data:str = "cifar10"):
    model = model.to(device)
    model.train()
    total_loss = 0
    pred_list = None
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    for i,batch in enumerate(dataloader):
        if modeltype != "clip":
            grad = True
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred,y_batch.reshape(-1).long())
            total_loss += loss.item()
            pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch
        
        else:
            grad =True # CLIP Model doesn't require gradients by default (no training required)
            optimizer.zero_grad()
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if data == "cifar10":
                captions = cifar10_label_to_text(y_batch.cpu().numpy())
            elif data == "pacs":
                captions = PACS_label_to_text(y_batch.cpu().numpy())
            encoded_captions = tokenizer(captions, padding=True, truncation=True, max_length = 200)
            encoded_captions =  {key: torch.tensor(values) for key, values in encoded_captions.items()}
            batch = {'image': x_batch, 'caption': encoded_captions}

            batch = move_tensors_to_device(batch,device)
            pred = model(batch)
            loss_i = loss_fn(pred, y_batch) 
            loss_t = loss_fn(pred.T, y_batch) 
            loss = (loss_i + loss_t)/2
            total_loss += loss.item()
            # print(torch.argmax(pred,dim=1),y_batch)
            pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

        # if i % (len(dataloader)//20) == 0 or i == len(dataloader)-1:
        #     print("Batch",i,":")
        #     print(f"Loss: {total_loss/(i+1):.4f}")
        #     accuracy = pred_list.float().mean().cpu().numpy()
        #     print(f"Accuracy: {accuracy:.4f}")

        if grad:
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

    if modeltype == 'clip':
        if data == "cifar10":        
            captions = cifar10_label_to_text([0,1,2,3,4,5,6,7,8,9])
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            encoded_captions = tokenizer(captions, padding=True, truncation=True, max_length = 200)
            encoded_captions =  {key: torch.tensor(values) for key, values in encoded_captions.items()}
        elif data == "pacs":
            captions = PACS_label_to_text([0,1,2,3,4,5,6])
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
            image, y_batch = batch
            image, y_batch = image.to(device), y_batch.to(device)
            batch = {'image': image, 'caption': encoded_captions}
            batch = move_tensors_to_device(batch,device)
            pred = model(batch)
            loss = loss_fn(pred, y_batch) 
            #print(torch.argmax(pred,dim=1),y_batch)
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

        total_loss += loss.item()
            
        # if i % (len(dataloader)//20) == 0 or i == len(dataloader)-1:
        #     print("Batch",i,":")
        #     print(f"Loss: {total_loss/(i+1):.4f}")
        #     accuracy = pred_list.float().mean().cpu().numpy()
        #     print(f"Accuracy: {accuracy:.4f}")

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
        image = self.images[idx]
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
    
    class_names = ['an airplane', 'an automobile', 'a bird', 'a cat', 'a deer', 'a dog', 'a frog', 'a horse', 'a ship', 'a truck']
    return [f"This is a photo of a {class_names[label]}." for label in labels]

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.data = images
        self.targets = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

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
                indices = range(len(class_images[0]))
                random.shuffle(indices)
                for i in range(indices):
                    for class_label in range(7):
                        shuffled_images.append(class_images[class_label][i])
                        shuffled_labels.append(class_label)
                train_dataset = CustomDataset(shuffled_images,shuffled_labels,transform=transforms_clip_vit)
                test_dataset = CustomDataset(test_dataset.data,test_dataset.targets,transform=transforms_clip_vit)
                self.train_loader = DataLoader(train_dataset,batch_size=10,shuffle=False)
                self.test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    def get_loaders(self):
        return self.train_loader, self.test_loader

#``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

class PACS_dataloaders(DataLoader):
    def __init__(
        self,model_type,test_domain,batch_size,shuffle
    ):
        domains = ['art_painting', 'cartoon','photo','sketch'] 
        self.test_domain = test_domain
        self.train_domain = [domain for domain in domains if domain != test_domain]
        data_dir = 'PACS'
        if model_type == 'resnet' or model_type == 'vit':
            transform = transforms_resnet if model_type == 'resnet' else transforms_clip_vit
            self.train_dataset = []
            for domain in self.train_domain:
                train_path = os.path.join(data_dir,domain)
                self.train_dataset.append(datasets.ImageFolder(train_path, transform=transform)) 
            self.train_dataset = ConcatDataset(self.train_dataset)
            test_path = os.path.join(data_dir,self.test_domain)
            self.test_dataset = datasets.ImageFolder(test_path, transform=transform)
            self.train_loader = DataLoader(self.train_dataset,batch_size=batch_size,shuffle=shuffle) 
            self.test_loader = DataLoader(self.test_dataset,batch_size=batch_size,shuffle=shuffle) 
        
        else:
            self.train_dataset = []
            for domain in self.train_domain:
                train_path = os.path.join(data_dir,domain)
                self.train_dataset.append(datasets.ImageFolder(train_path)) 
            self.train_dataset = ConcatDataset(self.train_dataset)
            test_path = os.path.join(data_dir,self.test_domain)
            self.test_dataset = datasets.ImageFolder(test_path)
            
            #print(self.test_dataset.class_to_idx)
            images = []
            labels = []
            for image, label in self.train_dataset:
               images.append(image)
               labels.append(label)

            class_images = {i: [] for i in range(7)}

            for img, label in zip(images, labels):
                class_images[label].append(img)

            shuffled_images = []
            shuffled_labels = []
            images = []
            labels = []
            for image, label in self.test_dataset:
               images.append(image)
               labels.append(label)


            min_samples_per_class = min([len(class_images[i]) for i in range(7)])
            samples = list(range(min_samples_per_class))
            random.shuffle(samples)

            for i in range(min_samples_per_class):
                for class_label in range(7):
                    shuffled_images.append(class_images[class_label][i])
                    shuffled_labels.append(class_label)
            train_dataset = CustomDataset(shuffled_images,shuffled_labels,transform=transforms_clip_vit)
            self.test_dataset = CustomDataset(images,labels,transform=transforms_clip_vit)
            self.train_loader = DataLoader(train_dataset,batch_size=7,shuffle=False)
            self.test_loader = DataLoader(self.test_dataset,batch_size=batch_size,shuffle=shuffle)
    
    def get_loaders(self):
        return self.train_loader, self.test_loader
            
def PACS_label_to_text(labels):
    
    class_names = ['a dog', 'an elephant', 'a giraffe', 'a guitar','a horse', 'a house', 'a person']
    return [f"This is an image of {class_names[label]}." for label in labels]
    
#``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

def CIFAR100_Splits(cifar100_images,cifar100_labels,group,modeltype):
    groups = {
    9: ["cattle", "shrew", "motorcycle", "squirrel", "snake", "trout", "sea", "tractor", "bus", "pickup"],
    8: ["bear", "elephant", "leopard", "camel", "lizard", "rabbit", "beaver", "spider", "raccoon", "orchid"],
    7: ["lion", "mountain", "crab", "bicycle", "turtle", "beetle", "train", "mouse", "snail", "otter"],
    6: ["possum", "shark", "forest", "pine", "dinosaur", "boy", "porcupine", "wolf", "road", "butterfly"], 
    5: ["girl", "rocket", "man", "tiger", "bee", "tank", "whale", "baby", "kangaroo", "dolphin"],
    4: ["willow", "worm", "chimpanzee", "skunk", "cup", "mushroom", "oak", "cockroach", "crocodile", "hamster"], 
    3: ["castle", "can", "bridge", "lobster", "house", "bed", "fox", "maple", "pear", "woman"], 
    2: ["palm", "streetcar", "pepper", "keyboard", "bottle", "seal", "rose", "couch", "caterpillar", "goldfish"], 
    1: ["flatfish", "apple", "orange", "plate", "table", "tulip", "bowl", "television", "skyscraper", "ray"], 
    0: ["wardrobe", "lamp", "plain", "lawnmower", "chair", "poppy", "clock", "cloud", "sunflower", "telephone"]
    }
    
    # cifar10_groups = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    cifar100_class_to_idx = {
    'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4,
    'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 
    'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 
    'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 
    'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 
    'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29, 
    'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 
    'girl': 35, 'hamster': 36, 'house': 37, 'kangaroo': 38, 'keyboard': 39, 
    'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 
    'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 
    'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 
    'otter': 55, 'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 
    'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 
    'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 
    'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 
    'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 
    'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 
    'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 
    'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93, 'wardrobe': 94, 
    'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99
    }

    transform = transforms_resnet if modeltype == "resnet" else transforms_clip_vit
    
    filtered_images, filtered_labels = [], []
    g = groups[group]
    g = [cifar100_class_to_idx[c] for c in g]
    for image, label in zip(cifar100_images,cifar100_labels):
        if label in g:
            l = g.index(label)
            filtered_images.append(image)
            filtered_labels.append(l)
    
    dataset = CustomDataset(filtered_images,filtered_labels,transform=transform)
    
    return dataset    


       
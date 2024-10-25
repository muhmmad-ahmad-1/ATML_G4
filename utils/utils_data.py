import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

mean = (0.5071, 0.4865, 0.4409)
std  = (0.2673, 0.2564, 0.2762)

BATCH_SIZE = 32

transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
		transforms.RandomCrop(32),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
transform_test = transform = transforms.Compose([
        transforms.CenterCrop(32),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean,std=std)
])

train = CIFAR100(root='data',train=True,transform=transform_train,download='False')
test = CIFAR100(root='data',train=False,transform=transform_test,download='False')

train_loader = DataLoader(train,batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test,batch_size=BATCH_SIZE, shuffle=False)
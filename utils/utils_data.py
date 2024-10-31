import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
import numpy as np
import PIL
from PIL import Image

mean = (0.5071, 0.4865, 0.4409)
std  = (0.2673, 0.2564, 0.2762)

BATCH_SIZE = 64

# transform_train = transforms.Compose([
#     transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),                          
#     transforms.RandomCrop(224, padding=4),            
#     transforms.RandomHorizontalFlip(),               
#     transforms.ToTensor(),                            
#     transforms.Normalize(mean,std)   
# ])

# transform_test = transforms.Compose([
#     transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),                           
#     transforms.CenterCrop(224),                          
#     transforms.ToTensor(),                            
#     transforms.Normalize(mean,std)   
# ])

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
])
transform_test = transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=mean,std=std)
])

# mean, std = 121.936 / 255, 68.389 / 255  # Normalizing within 0-1 range

# transform_train = transforms.Compose([
#     transforms.RandomRotation(15),            # Rotation up to 15 degrees
#     transforms.RandomHorizontalFlip(),         # Horizontal flipping
#     transforms.RandomResizedCrop(size=32, scale=(0.9, 1.1)),  # Width and height shifts, similar to (0.1, 0.1) scaling
#     transforms.ToTensor(),
#     transforms.Normalize(mean,std),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(mean,), std=(std,)),
# ])

train_data = CIFAR100(root='data',train=True,transform=transform_train,download='False')
test_data = CIFAR100(root='data',train=False,transform=transform_test,download='False')

train_loader = DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data,batch_size=BATCH_SIZE, shuffle=False)

class CIFAR100IdxSample(CIFAR100):
	def __init__(self, root, train=True, 
				 transform=None, target_transform=None,
				 download=False, n=4096, mode='exact', percent=1.0):
		super().__init__(root=root, train=train, download=download,
						 transform=transform, target_transform=target_transform)
		self.n = n
		self.mode = mode

		num_classes = 100
		num_samples = len(self.data)
		labels = self.targets

		self.cls_positive = [[] for _ in range(num_classes)]
		for i in range(num_samples):
			self.cls_positive[labels[i]].append(i)

		self.cls_negative = [[] for _ in range(num_classes)]
		for i in range(num_classes):
			for j in range(num_classes):
				if j == i:
					continue
				self.cls_negative[i].extend(self.cls_positive[j])

		self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
		self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

		if 0 < percent < 1:
			num = int(len(self.cls_negative[0]) * percent)
			self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:num]
								 for i in range(num_classes)]

		self.cls_positive = np.asarray(self.cls_positive)
		self.cls_negative = np.asarray(self.cls_negative)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		if self.mode == 'exact':
			pos_idx = index
		elif self.mode == 'relax':
			pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
		else:
			raise NotImplementedError(self.mode)
		replace = True if self.n > len(self.cls_negative[target]) else False
		neg_idx = np.random.choice(self.cls_negative[target], self.n, replace=replace)
		sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

		return img, target, index, sample_idx

train_idx =  CIFAR100IdxSample(root='data',train=True,transform=transform_train,download='False')
test_idx =  CIFAR100IdxSample(root='data',train=False,transform=transform_test,download='False')

train_loader_idx = DataLoader(train_idx,batch_size=BATCH_SIZE, shuffle=True)
test_loader_idx = DataLoader(test_idx,batch_size=BATCH_SIZE, shuffle=False)

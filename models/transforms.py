from torchvision import transforms
import numpy as np
import torch

transforms_resnet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

transforms_vit = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transforms_clip_vit = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

class RandomNoise(torch.nn.Module):
    def __init__(self,patch_size=8,mean=0.0,std=1.0):
        super().__init__()
        self.patch_size = patch_size
        noise = np.random.normal(mean,std,(self.patch_size,self.patch_size,3))
        noise = 128* noise + 127
        self.noise = np.array(noise,dtype='uint8')
        self.random_patch = np.random.randint(1,32-self.patch_size+1,(2,))
    def forward(self, img):  # we assume inputs are always structured like this
        image  = np.array(img)
        image[self.random_patch[0]:self.random_patch[0]+self.patch_size,self.random_patch[1]:self.random_patch[1]+self.patch_size,:] += self.noise
        image[image>255] = 255
        image[image<0] = 0
        return image

class RandomScramble(torch.nn.Module):
    def __init__(self,patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.patch_num = 32//self.patch_size
    def forward(self, image):  # we assume inputs are always structured like this
        img = np.array(image)
        patches = img.reshape(self.patch_num, self.patch_size, self.patch_num, self.patch_size, 3).swapaxes(1, 2).reshape(-1, self.patch_size, self.patch_size, 3)
        np.random.shuffle(patches)
        scrambled_image = patches.reshape(self.patch_num, self.patch_num, self.patch_size, self.patch_size, 3).swapaxes(1, 2).reshape(32, 32, 3)
        return scrambled_image

class RandomColorChange(torch.nn.Module):
    def forward(self, image):  # we assume inputs are always structured like this
        rand_num = np.random.randint(0,256,3)
        img = np.array(image)
        for i in range(3):
            img[:,:,i][img[:,:,i]>rand_num[i]] = img[:,:,i][img[:,:,i]>rand_num[i]] - rand_num[i]
            img[:,:,i][img[:,:,i]<=rand_num[i]] = rand_num[i] - img[:,:,i][img[:,:,i]<=rand_num[i]]
        img = np.abs(img)
        return img
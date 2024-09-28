from torchvision import transforms
import numpy as np
import torch
from skimage import color, io, feature
from skimage.feature import canny
import cv2

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

class Silhouette(torch.nn.Module):
    def forward(self,image):
        grayscale = color.rgb2gray(image)
        mean_value = np.mean(grayscale)
        _, binarized_image = cv2.threshold(grayscale, mean_value, 255, cv2.THRESH_BINARY)
        binarized_image = binarized_image.reshape(1,96, 96)
        binarized_image = np.repeat(binarized_image, 3, axis=0).transpose(1,2,0)
        return binarized_image.astype(np.float32)

class CleanerSilhouette(torch.nn.Module):
    def forward(self,image):
        grayscale = color.rgb2gray(image)
        mean_value = np.mean(grayscale)
        _, binarized_image = cv2.threshold(grayscale, mean_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        cleaned_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel)
        cleaned_image = cleaned_image.reshape(1,96, 96)
        cleaned_image = np.repeat(cleaned_image, 3, axis=0).transpose(1,2,0)
        return cleaned_image.astype(np.float32)

class Grayscale(torch.nn.Module):
    def forward(self,image):
        grayscale = color.rgb2gray(image)
        grayscale = grayscale.reshape(1,96, 96)
        grayscale = np.repeat(grayscale, 3, axis=0).transpose(1,2,0)
        return grayscale.astype(np.float32)

class Blur(torch.nn.Module):
    def forward(self,image):
        blurred_image = cv2.GaussianBlur(image, (21, 21), sigmaX=0, sigmaY=0)
        blurred_image = cv2.GaussianBlur(blurred_image, (11, 11), sigmaX=0, sigmaY=0)
        blurred_image = cv2.GaussianBlur(blurred_image, (5, 5), sigmaX=0, sigmaY=0)
        
        downsampled = cv2.resize(blurred_image, (12,12), interpolation=cv2.INTER_LINEAR)
        
        upsampled = cv2.resize(downsampled, (96, 96), interpolation=cv2.INTER_LINEAR)
        
        upsampled = cv2.GaussianBlur(upsampled, (21, 21), sigmaX=0, sigmaY=0)

        return upsampled

class Outline(torch.nn.Module):
    def forward(self,image):
        gray = color.rgb2gray(image)
        edges = canny(gray, sigma=0.8,low_threshold=0.15,high_threshold=0.5)
        edges = edges.reshape(1,96, 96)
        edges = np.repeat(edges, 3, axis=0).transpose(1,2,0)
        return edges.astype(np.float32)

class RandomPatchTexture(torch.nn.Module):
    def forward(self,image):
        rows, cols = image.shape[:2]
        scale_range=(1.5, 2); rotation_range=(-90, 90)
        
        scale = np.random.uniform(scale_range[0], scale_range[1])
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        
        # Get transformation matrix for rotation and scaling
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, scale)
        
        # Apply the affine transformation
        transformed_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT_101)
        
        return transformed_image

class DistortedTexture(torch.nn.Module):
    def forward(self,image):
        random_state = np.random.RandomState(None)
        sigma = 1
        alpha = 15
        shape = image.shape
        # Generate random displacement fields
        dx = cv2.GaussianBlur((random_state.rand(*shape[:2]) * 2 - 1).astype(np.float32), (5, 5), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape[:2]) * 2 - 1).astype(np.float32), (5, 5), sigma) * alpha

        # Generate meshgrid coordinates
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Add displacement fields to meshgrid coordinates
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Apply the elastic transformation using cv2.remap
        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) 
        
        return distorted_image
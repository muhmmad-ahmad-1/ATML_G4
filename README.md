# Assignment 1 - Deep Models and Domain Generalization

## Task 0: Model Selection
* Discriminative Models:
1) ResNet-18
2) ResNet-50
3) ViT-Base_Patch16_224

* Contrastive Models:
1) ViT_Base_Patch16_CLIP_224 w/ Distilbert-Base-Uncased

* Diffusion Zero-Shot Classifier (for Task 1):
1) stable-diffusion-v1-4

The complete implementation of each of the models with the necessary utility functions can be found [here](models/model.py).

## Task 1: Text-to-Image Generative Model as Zero-Shot Classifier
A pipeline was developed for zero-shot image classification in which an input image and a predefined list of classes are processed. The method involves transforming each class into a corresponding text prompt, followed by converting the input image into its latent representation. 
The latent image is put through a process of noise, after which an iterative process of denoising is performed for N iterations. During each iteration a random starting point for denoising is chosen and the image is denoised conditioned on all prompts, resulting in a series of N reconstructed images for each class/prompt.

Following the reconstruction process, the norm-squared of the difference between each class-specific reconstructed image and the original image is calculated, producing a score matrix. This score matrix is then multiplied by a standard weight vector, and the argmin class label is returned.
## Task 2: Evaluation on an IID Dataset
| Model             | Epochs Trained | Top-1 Accuracy (Train) | Top-1 Accuracy (Test)
| :---------------- | :------: | :----: | :---: |
| ResNet-18        |   30   | 79.05 | 78.21 | 
| ResNet-50           |   3   | 78.67 | 78.01 |
| ViT-Base_Patch16_224   |  3   | 94.49 | 94.44 |
| ViT_Base_Patch16_CLIP_224 |  3   | 95.44 | 95.41 |

## Task 3: Evaluation for Domain Generalization
* PCAS Dataset: All Models are trained on 3 epochs (Covariate Shift)

| Model                  | Art  | Cartoon    | Photo | Sketch |
| :--------------------- | :----: | :----: | :-----: | :----: |
| ResNet-50              | 63.82  | 53.11  | 92.57   | 49.45  |
| ViT-Base_Patch16_224   | 88.67  | 77.39  | 90.36   | 64.52  |
| ViT_Base_Patch16_CLIP_224 | 85.93  | 69.84  | 98.44   | 70.40  |

* SVHN: All models trained on 3 epochs (Concept/Semantic Shift)

| Model             | Epochs Trained | Top-1 Accuracy (Train) | Top-1 Accuracy (Test)
| :---------------- | :------: | :----: | :---: |
| ResNet-50           |   3   | 49.75 | 50.82 |
| ViT-Base_Patch16_224   |  3   | 58.10 | 61.84 |
| ViT_Base_Patch16_CLIP_224 |  3   | 41.16 | 56.25 |

## Task 4: Inductive Biases of Models: Semantic Biases
### Datasets Creation:
STL-10 dataset:
airplane, bird, car, cat, dog, ship, truck,deer, horse,monkey

About the Dataset: STL-10 is a subset of TinyImageNet just like CIFAR-10 but has a higher resolution

#### Shape Bias Dataset Creation
1) Image Processing Techniques to get Silhouette and Outline
#### Texture Bias Dataset Creation
1) Texture Extraction with Shape Removal/Distortion with Techniques like Random Affine Transformations for Patch Focusing or Distorting Filters

#### Color Bias Dataset Creation
1) Colored MNIST (Directly Used)
2) Color retained STL-10 with outline removal through blurring (Gaussian filter)

### Results

## Task 5: Inductive Biases of Models: Locality Biases 
### Datasets Creation (CIFAR10 Modification):
#### Localized Noise Injection
For each image in the test dataset, the same 8x8 patch was selected and random noise was added to this patch. The motivation behind the 8x8 patch for change was the fact that the ResNet50 model has a kernel of 7x7 in its first convolutional layer. Our idea was to intorduce a change that would modify the values of a large portion of the feature maps resulting after the first convolutional layer significantly and introduce similar effects downstream.

#### Global Style Changes
We introduce this in two ways:
* A simple random color shift:
Each channel is perturbed (subtraction) by a random integer between 0 and 255 in all of its pixel values, taking the absolute value of the resultant pixel values to allow for negative perturbations.

* A style shift through a VGG Model 
(TODO)


#### Scrambled Images
Each image is divided into 16 patches of 8x8(x3) and the patches are randomly permuted.

Each of these (apart from the style change) is implemented as a custom transform which can be joined with typical transforms from the "transforms" library from torchvision. Consequently, a complete presentation of the resultant dataset is redundant. We do however show some examples of the resultant images after transform [here](http://insertdrivelink.com). The complete style transfer dataset can be seen [here](http://insertdrivelink.com).

### Results
All results are of test acccuracies:
| Model             | Raw  | Noise  | Color  | Style | Scramble  |
| :---------------- | :------: | :----: | :---: | :---: | :---: |
| ResNet-50           |   78.01  | 65.87 | 27.53 | | 35.86 |
| ViT-Base_Patch16_224   |  94.44  | 91.28  | 62.40 | | 65.56 |
| ViT_Base_Patch16_CLIP_224 |  97.11  | 96.59 | 48.10 | | 45.35 |






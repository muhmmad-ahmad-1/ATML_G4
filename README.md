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

## Task 2: Evaluation on an IID Dataset
| Model             | Epochs Trained | Top-1 Accuracy (Train) | Top-1 Accuracy (Test)
| :---------------- | :------: | :----: | :---:
| ResNet-18        |   30   | 79.05 | 78.21 | 
| ResNet-50           |   3   | 78.67 | 78.01 |
| ViT-Base_Patch16_224   |  3   | 94.49 | 94.44 |
| ViT_Base_Patch16_CLIP_224 |  3   | 95.44 | 95.41 |

## Task 3: Evaluation for Domain Generalization
* PCAS Dataset: 


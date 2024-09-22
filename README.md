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
| :---------------- | :------: | :----: | :---: |
| ResNet-18        |   30   | 79.05 | 78.21 | 
| ResNet-50           |   3   | 78.67 | 78.01 |
| ViT-Base_Patch16_224   |  3   | 94.49 | 94.44 |
| ViT_Base_Patch16_CLIP_224 |  3   | 95.44 | 95.41 |

## Task 3: Evaluation for Domain Generalization
* PCAS Dataset: All Models are trained on 3 epochs (Covariate Shift)

| Model                  | Photo  | Art    | Cartoon | Sketch |
| :--------------------- | :----: | :----: | :-----: | :----: |
| ResNet-50              | 63.82  | 53.11  | 92.57   | 49.45  |
| ViT-Base_Patch16_224   | 88.67  | 77.39  | 90.36   | 64.52  |
| ViT_Base_Patch16_CLIP_224 | 85.93  | 69.84  | 98.44   | 70.40  |

* CIFAR100 Splits: All models trained on 3 epochs (CIFAR10) (Concept/Semantic Shift)
Test the models on each of the 10 splits of CIFAR100 with lower split number indicating greater concept shift
No finetuning is done on CIFAR100 dataset

| Model                  |  0  |  1 |  2  |  3  |  4  |  5 |  6  |  7 |  8  | 9 |
| :--------------------- | :----: | :----: | :-----: | :----: | :----: | :----: | :-----: | :----: | :-----: | :----: |
| ResNet-50              | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ViT-Base_Patch16_224   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ViT_Base_Patch16_CLIP_224 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |




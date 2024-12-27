import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
from utils.utils_eval import Test
import logging
from utils.utils_metric import Gradient_Metrics
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



import torch.nn.functional as F

class DistillationLoss:
    def __init__(self, T=4):
        self.T = T

    def __call__(self, student, teacher):
        # Apply temperature scaling
        student = F.log_softmax(student / self.T, dim=-1)  # Log-softmax on student with temperature
        teacher = F.softmax(teacher / self.T, dim=-1)      # Softmax on teacher with temperature

        # Compute the KL Divergence between the teacher and student
        loss = F.kl_div(student, teacher, reduction='batchmean')  # KL Divergence

        # Scale by temperature squared (this is common in distillation loss)
        return loss * (self.T ** 2)

def get_files_in_directory(directory):
    return glob.glob(f"{directory}/*")

# Dataset Class for Structured Noise
class StructuredNoiseDataset(Dataset):
    def __init__(self, image_paths, transform=None, dataset_name=None):
        self.image_paths = get_files_in_directory(image_paths)
        self.transform = transform
        self.dataset_name = dataset_name
        self.image = []  
        for image_path in self.image_paths:
            with Image.open(image_path) as image:  
                image = np.array(image)
                self.image.append(image)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]

        # Apply dataset-specific preprocessing
        if self.dataset_name == "MNIST":
            image = Image.fromarray(image)
            image = image.convert("L")  # Convert to grayscale
            image = image.resize((32, 32))  # Resize to 32x32
            image = np.stack([image] * 3, axis=-1)  # Repeat grayscale channel 3 times
            image = Image.fromarray(image)

        elif self.dataset_name == "CIFAR10":
            image = Image.fromarray(image)
            image = image.resize((32, 32))  # Resize to 32x32
            image = np.array(image)
            image = image[:, :, :3]  # Remove alpha channel if present
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        return image

#dataset_mnist = StructuredNoiseDataset("Data\small_scale\dead_leaves-squares\Final", dataset_name="MNIST", transform=transforms.ToTensor())
dataset_cifar10 = StructuredNoiseDataset("Data\small_scale\dead_leaves-squares\Final", dataset_name="CIFAR10", transform=transforms.ToTensor())
dataset_mnist = 0

# Utility Functions
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy_metric(outputs, targets):
    return np.mean(outputs.detach().cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def weight_reset(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
        m.reset_parameters()


def calculate_reliability(logits, lambd1, lambd2):
    probs = torch.nn.functional.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
    diversity = -torch.mean(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-10))
    return 1 / (entropy * lambd1 - diversity * lambd2 + 1e-10)


# NEKD Class
class NEKD:
    def __init__(self, teachers, student, lambd_1, lambd_2, n_epochs, lr, n_batches, batch_size, dataset_name):
        self.teachers = teachers
        self.student = student
        self.lambd1 = lambd_1
        self.lambd2 = lambd_2
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.device = get_device()

        # Transform based on dataset
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # Dataset and DataLoader
        dataset = dataset_mnist if dataset_name == "MNIST" else dataset_cifar10
        dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.dataloader

    def train(self):
        criterion = DistillationLoss()
        self.student.train()
        for teacher in self.teachers:
            teacher.train()
        print("\nNEKD Training Started...")
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        train_loss = 0
        avg_reliability = torch.zeros((1, len(self.teachers))).to(self.device)
        iterator = iter(self.dataloader)
        for epoch in range(self.n_epochs):
            train_loss_1 = train_loss
            for i in range(self.n_batches):
                try:
                    data = next(iterator)
                except StopIteration:
                    print("Data iterator reset")
                    iterator = iter(self.dataloader)
                    data = next(iterator)
                optimizer.zero_grad()
                data = data.to(self.device)

                with torch.no_grad():
                    teacher_output = [teacher(data) for teacher in self.teachers]

                student_output = self.student(data)

                # Calculate reliability
                reliability = torch.stack([
                    calculate_reliability(teacher_out, self.lambd1, self.lambd2)
                    for teacher_out in teacher_output
                ]).to(self.device)
                reliability = reliability / torch.sum(reliability)  # Normalize reliability
                reliability = torch.nn.functional.softmax(reliability, dim=0)

                # Calculate losses
                loss = torch.stack([
                    criterion(student_output, teacher_out)
                    for teacher_out in teacher_output
                ])

                # Weighted loss
                weighted_loss = torch.sum(loss * reliability)

                # Backpropagation
                weighted_loss.backward()
                optimizer.step()

                train_loss += weighted_loss.item()
                avg_reliability += reliability
            train_loss_1 = train_loss - train_loss_1
            logging.log(logging.INFO, f"Epoch: {epoch + 1}, Loss: {train_loss_1}")
            metrics = Gradient_Metrics(10,[{name:p.grad for name,p in teacher.named_parameters()} for teacher in self.teachers],{name:p.grad for name,p in self.student.named_parameters()})
            metrics.calculate_metrics()
            metrics = metrics.metrics
            logging.info(f"Cosine Similarity: {metrics['Average Cosine Similarity (Global Grad)']}, Variance: {metrics['Variance Grad']}")

        avg_reliability /= (self.n_batches * self.n_epochs)
        del self.dataloader
        return self.student, avg_reliability, train_loss / (self.n_batches * self.n_epochs * self.batch_size)

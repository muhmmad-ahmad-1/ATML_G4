import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import glob


# Distillation Loss Class
class DistillationLoss:
    def __init__(self, T=4):
        self.T = T

    def __call__(self, student, teacher):
        student = F.log_softmax(student / self.T, dim=-1)
        teacher = (teacher / self.T).softmax(dim=-1)

        try:
            loss = -(teacher * student).sum(dim=1).mean() 
            loss = loss*(self.T)**2
            return loss
        except:
            import pdb;
            pdb.set_trace()

def get_files_in_directory(directory):
    return glob.glob(f"{directory}/*")

# Dataset Class for Structured Noise
class StructuredNoiseDataset(torch.utils.data.Dataset):
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
# dataset_cifar10 = None
dataset_cifar10 = StructuredNoiseDataset("Data\small_scale\dead_leaves-squares\Final", dataset_name="CIFAR10", transform=transforms.ToTensor())
dataset_mnist = None

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

class NEKD():
    def __init__(self, teachers, student, lambd_1, lambd_2, n_epochs, lr, train_loader, alpha1, alpha2, beta,dataset_name):
        self.teachers = teachers
        self.student = student
        self.lambd1 = lambd_1
        self.lambd2 = lambd_2
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.device = get_device()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.lr = lr
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # Dataset and DataLoader
        dataset = dataset_mnist if dataset_name == "MNIST" else dataset_cifar10
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.train_loader.batch_size, shuffle=True)

    def train(self):
        criterion = DistillationLoss()
        xe = nn.CrossEntropyLoss(reduction='mean')
        self.student.to(self.device)
        self.student.train()
        for teacher in self.teachers:
            teacher.to(self.device)
            teacher.train()
        print("\nNEKD.....")
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        train_loss = 0
        avg_reliability = torch.zeros((1, len(self.teachers))).to(self.device)
        for epoch in range(self.n_epochs):
            data_iterator = iter(self.dataloader)
            for x_train, y_train in self.train_loader:
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                try:
                    x_noise = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.dataloader)
                    x_noise = next(data_iterator)
                x_noise = x_noise.to(self.device)
                noise = torch.randn_like(x_train, device=self.device)
                optimizer.zero_grad()

                teacher_gold = [teacher(x_train) for teacher in self.teachers]
                student_gold = self.student(x_train)

                # Calculate reliability
                reliability = torch.stack(
                    [calculate_reliability(teacher_out, self.lambd1, self.lambd2) for teacher_out in teacher_gold]).to(
                    self.device)
                reliability = reliability / torch.sum(reliability)  # Normalize reliability
                reliability = torch.nn.functional.softmax(reliability, dim=0)
                reliability = reliability.detach()

                # Calculate losses
                l_kd_b = torch.stack([criterion(student_gold, teacher_out) for teacher_out in teacher_gold])
                l_xe = xe(student_gold, y_train.reshape(-1).long())
                weighted_loss_b = torch.sum(l_kd_b * reliability) * self.alpha2
                loss_xe = l_xe * self.beta
                gold_loss = weighted_loss_b + loss_xe

                gold_loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                # Weighted loss
                teacher_noise = [teacher(noise) for teacher in self.teachers]
                student_noise = self.student(noise)
                l_kd_g = torch.stack([criterion(student_noise, teacher_out) for teacher_out in teacher_noise])
                weighted_loss_g = torch.sum(l_kd_g * reliability) * self.alpha1

                # Backpropagation
                weighted_loss_g.backward()
                optimizer.step()

                train_loss += (weighted_loss_g + gold_loss).item()
                avg_reliability += reliability

        avg_reliability /= (len(self.train_loader) * self.n_epochs)
        return self.student, avg_reliability, train_loss / (
                    len(self.train_loader) * self.n_epochs * student_gold.shape[0])





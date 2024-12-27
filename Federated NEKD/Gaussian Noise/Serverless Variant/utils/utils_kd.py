import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models import ResNet18
from utils.utils_metric import Gradient_Metrics
from utils.utils_eval import Test
import logging


logging.basicConfig(filename='app_noise.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



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

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy_metric(outputs, targets):
    return np.mean(outputs.detach().cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or \
        isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()

def calculate_reliability(logits,lambd1,lambd2):
    probs = torch.nn.functional.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
    diversity = -torch.mean(probs.mean(dim=0) * torch.log(probs.mean(dim=0) + 1e-10))
    return 1 / (entropy*lambd1 - diversity*lambd2 + 1e-10)

def model_to_vec(model):    
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec).to(get_device())  

class NEKD():
    def __init__(self,teachers,student, lambd_1, lambd_2, n_epochs, lr,train_loader,alpha1,alpha2,beta):
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
    
    def train(self):
        drift_penalty = 0.5
        original_student = ResNet18()
        original_student.load_state_dict(self.student.state_dict())
        original_student.to(self.device)
        criterion = DistillationLoss()
        xe = nn.CrossEntropyLoss(reduction='mean')
        self.student.to(self.device)
        self.student.train()
        for teacher in self.teachers:
            teacher.to(self.device)
            teacher.train()
        print("\nNEKD.....")
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        train_loss_1 = 0
        ce_loss = 0
        distill_loss = 0
        avg_reliability = torch.zeros((1,len(self.teachers))).to(self.device)
        for epoch in range(self.n_epochs):
            train_loss = 0
            ce_loss = 0
            distill_loss = 0
            logging.info(f"Distillation Epoch: {epoch + 1}")
            for x_train,y_train in self.train_loader:
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                noise = torch.randn_like(x_train, device=self.device)
                optimizer.zero_grad()
        
                teacher_gold = [teacher(x_train) for teacher in self.teachers]
                student_gold = self.student(x_train)
                
        
                # Calculate reliability
                reliability = torch.stack([calculate_reliability(teacher_out, self.lambd1, self.lambd2) for teacher_out in teacher_gold]).to(self.device)
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
                weighted_loss_g = torch.mean(l_kd_g) * self.alpha1
                
                #Backpropagation
                weighted_loss_g.backward()
                optimizer.step()
        
                train_loss += (weighted_loss_g + gold_loss).item()
                #train_loss += (gold_loss).item()
                avg_reliability += reliability
                ce_loss += l_xe.item()
                distill_loss += (weighted_loss_g+weighted_loss_b).item()
            train_loss_1 += train_loss / (len(self.train_loader))
            logging.info(f"Epoch: {epoch + 1} | Train Loss: {train_loss / (len(self.train_loader))} | CE Loss: {ce_loss / (len(self.train_loader))} | Distill Loss: {distill_loss / (len(self.train_loader))}")
            metrics = Gradient_Metrics(10,[{name:p.grad for name,p in teacher.named_parameters()} for teacher in self.teachers],{name:p.grad for name,p in self.student.named_parameters()})
            metrics.calculate_metrics()
            metrics = metrics.metrics
            logging.info(f"Cosine Similarity: {metrics['Average Cosine Similarity (Global Grad)']}, Variance: {metrics['Variance Grad']}")

        
        avg_reliability /= (len(self.train_loader) * self.n_epochs)
        
        return self.student, avg_reliability, train_loss_1 / (self.n_epochs * student_gold.shape[0])





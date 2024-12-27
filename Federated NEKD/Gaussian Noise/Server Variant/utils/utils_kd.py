import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class DistillationLoss:
    def __init__(self, T=4):
        self.T = T

    def __call__(self, student, teacher):
        student = F.log_softmax(student/self.T, dim=-1)
        teacher = (teacher/self.T).softmax(dim=-1)
        
        try: return -(teacher * student).sum(dim=1).mean()*(self.T)**2
        except: import pdb; pdb.set_trace()

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

class NEKD():
    def __init__(self,teachers,student, lambd_1, lambd_2, n_epochs, lr, n_batches, batch_size):
        self.teachers = teachers
        self.student = student
        self.lambd1 = lambd_1
        self.lambd2 = lambd_2
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_batches = n_batches
        self.sample = torch.zeros((batch_size,3,32,32))
        self.device = get_device()
    
    def train(self):
        criterion = DistillationLoss()
        xe = nn.CrossEntropyLoss(reduction='mean')
        self.student.train()
        for teacher in self.teachers:
            teacher.train()
        print("\nNEKD.....")
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        train_loss = 0
        avg_reliability = torch.zeros((1,len(self.teachers))).to(self.device)
        for epoch in range(self.n_epochs):
            for batch in range(self.n_batches):
                optimizer.zero_grad()
                data = torch.randn_like(self.sample, device=get_device())
        
                with torch.no_grad():
                    teacher_output = [teacher(data) for teacher in self.teachers]
        
                student_output = self.student(data)
        
                # Calculate reliability
                reliability = torch.stack([calculate_reliability(teacher_out, self.lambd1, self.lambd2) for teacher_out in teacher_output]).to(self.device)
                reliability = reliability / torch.sum(reliability)  # Normalize reliability
                reliability = torch.nn.functional.softmax(reliability, dim=0)
        
                # Calculate losses
                loss = torch.stack([criterion(student_output, teacher_out) for teacher_out in teacher_output])
        
                # Weighted loss
                weighted_loss = torch.sum(loss * reliability)
        
                # Backpropagation
                weighted_loss.backward()
                optimizer.step()
        
                train_loss += weighted_loss.item()
                avg_reliability += reliability
        
        avg_reliability /= (self.n_batches * self.n_epochs)
        return self.student, avg_reliability, train_loss / (self.n_batches * self.n_epochs * self.sample.shape[0])





from models import *
from utils.utils_dataset import *
from utils.utils_libs import *
import torch
from optimizers import SAM, ESAM
import copy
from torch.utils import data
from utils.utils_kd import NEKD
# from master import Args

class Client:
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        if dataset == "CIFAR10" or dataset == "MNIST":
            self.model = ResNet18()
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.loss_func = loss_func
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.max_norm = max_norm
            self.args = args
            self.deltas = None
            self.data_name = dataset
            self.dataset = Dataset(self.trn_x,self.trn_y,True,dataset)
            self.dataloader = data.DataLoader(self.dataset,batch_size=batch_size,shuffle=True)
            self.grad_aggregation = grad_aggregator
            self.grad = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
            self.epochs = epochs
            self.init_optimizer(optimizer,learning_rate,weight_decay)
            self.train_loss = torch.zeros((self.args.tr_rounds,))
            self.r = 0
            
        else:
            raise NotImplementedError("Invalid Dataset")

    def init_optimizer(self,optimizer,learning_rate,weight_decay):
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),learning_rate,weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),learning_rate,weight_decay=weight_decay)
    
    def train(self):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = ResNet18()
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train(); self.model = self.model.to(self.device)
        
        self.model.zero_grad()

        train_loss = 0
        
        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                
                y_pred = self.model(batch_x)
                
                loss = self.loss_func(y_pred, batch_y.reshape(-1).long())
                
                train_loss += loss.item()
                    
                loss.backward()

                self.optimizer.step()   
        
        train_loss /= len(self.dataloader) * self.epochs
        self.train_loss[self.r] = train_loss 
        self.r += 1     
        self.grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}
        
                
    def update_parameters(self, parameter_dict):
        for params_c,params_global in zip(self.model.parameters(),parameter_dict):
            params_c.data = params_global.clone()
    
    def distill(self,args,teacher_models,student_model=None):
        
        distill_dataloader= data.DataLoader(self.dataset,batch_size=args.distill_batch_size,shuffle=True)
        student_model.train()
        for teacher in teacher_models:
            teacher.train()
            
        student_model,reliability,distill_loss = NEKD(teacher_models,student_model, args.lambd1, args.lambd2, args.distill_epochs, args.distill_lr,distill_dataloader,args.alpha1,args.alpha2,args.beta).train()
        
        
        return student_model, reliability, distill_loss
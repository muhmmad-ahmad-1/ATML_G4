from models import *
from utils.utils_dataset import *
from utils.utils_libs import *
import torch
from optimizers import SAM, ESAM
import copy
from torch.utils import data

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

        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                
                y_pred = self.model(batch_x)
                
                loss = self.loss_func(y_pred, batch_y.reshape(-1).long())
                    
                loss.backward()

                self.optimizer.step()   
                
        self.grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}
        
                
    def update_parameters(self, parameter_dict):
        for params_c,params_global in zip(self.model.parameters(),parameter_dict):
            params_c.data = params_global.clone()
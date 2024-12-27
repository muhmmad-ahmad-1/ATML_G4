import torch
from client import *
from .server import Server,Args
import numpy as np
from utils.utils_kd import *
import pandas as pd
import copy
from models import ResNet18

class NEKD_FedSAM_Avg(Server):
    def __init__(self,args : Args, random_init :bool = False , aggregation:str = "default"):
        super(NEKD_FedSAM_Avg,self).__init__(args,random_init,aggregation)
        self.reliability = torch.zeros((self.args.tr_rounds,self.n_clients))
    def initialize_clients(self):
         self.clients = [fedsam(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def run_experiment(self):
        
        self.initialize_clients()
        
        print("Data Distributed Among Clients")

        self.update_params()
        
        print("Model Parameters Initialized")

        self.train_and_eval()
        
        self.results = pd.DataFrame.from_dict(self.metrics)
        self.results.to_csv("Results/"+self.data.name+"_"+self.aggregation+"_"+"_"+self.aggregator+"_"+self.opt_name+".csv")
        df = pd.DataFrame(self.reliability.numpy(), columns=[f'R_{i+1} ' for i in range(self.n_clients)])

        # Add an epoch index (optional)
        df.index.name = 'Epoch'
        df.to_csv("Results/"+self.data.name+"_"+self.aggregation+"_"+"_"+self.aggregator+"_"+self.opt_name+"_reliability.csv")
        
    def aggregate_grad(self):
        self.global_model = self.global_model.to(self.device)
        aggregated_params = [torch.zeros_like(param) for param in self.clients[0].model.parameters()]
        self.old_global_model = ResNet18()
        self.old_global_model = self.old_global_model.to(self.device)
        self.old_global_model.load_state_dict(self.global_model.state_dict())
        for j,client in enumerate(self.clients):
                for i, grad in enumerate(client.model.parameters()):
                    aggregated_params[i] += grad * self.data_ratio[j]
                    
        with torch.no_grad():
            for param, aggregated_param in zip(self.global_model.parameters(),aggregated_params):
                param.data.copy_(aggregated_param)
        # Aggregate normally
        # Then distill each client for rectification
        teachers = [client.model.to(self.device) for client in self.clients]
        
        lambd1 = self.args.lambd1
        lambd2 = self.args.lambd2
        distillation_epochs = self.args.distill_epochs
        distillation_lr = self.args.distill_lr
        
        distiller =NEKD(teachers,self.global_model,lambd1,lambd2,distillation_epochs,distillation_lr,100,256,self.dataset)
        
        new_student,reliability,train_loss = distiller.train()
        self.global_model.load_state_dict(new_student.state_dict())
        self.global_grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.global_model.named_parameters(),self.old_global_model.parameters())}
        self.metrics[self.r] = {"Distillation Loss": train_loss}
        self.reliability[self.r] = reliability.unsqueeze(0)
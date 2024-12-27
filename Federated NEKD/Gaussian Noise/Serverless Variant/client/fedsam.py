from .client import Client
import copy
from models import ResNet18
from optimizers.ESAM import ESAM
from utils import *

class fedsam(Client):
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        super(fedsam,self).__init__(dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator,args,epochs)
    
    
    def init_optimizer(self, optimizer, learning_rate, weight_decay):
        self.base_optimizer = torch.optim.SGD(self.model.parameters(),learning_rate,weight_decay=weight_decay)
        self.optimizer = ESAM(self.model.parameters(),self.base_optimizer,rho=self.args.rho,adaptive=True)
    
    
    def train(self):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = ResNet18().to(self.device)
        global_model.load_state_dict(global_model_params)
        global_model.to(self.device)
            
        self.model.train(); self.model = self.model.to(self.device)
        
        self.model.zero_grad()
        train_loss = 0
        for _ in range(self.epochs):
            for batch_x, batch_y in self.dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                self.optimizer.paras = [batch_x,batch_y,self.loss_func,self.model]
                loss = self.loss_func(self.model(batch_x),batch_y.reshape(-1).long())
                self.optimizer.step()
                
                torch.nn.utils.clip_grad_norm(parameters=self.model.parameters(),max_norm=self.max_norm)

                self.base_optimizer.step()   
                train_loss += loss.item()
        
        train_loss /= len(self.dataloader) * self.epochs
                
        self.train_loss[self.r] = train_loss 
        self.r += 1  
                
        self.grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}
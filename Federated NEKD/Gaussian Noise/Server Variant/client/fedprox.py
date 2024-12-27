from .client import Client
import copy
from models import *

class fedprox(Client):
    def __init__(self,dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator = False,args = {"mu":0.0},epochs=3):
        super(fedprox,self).__init__(dataset,trn_x,trn_y,batch_size,loss_func,learning_rate,weight_decay,optimizer,max_norm,grad_aggregator,args,epochs)
    
    def train(self):
        global_model_params = copy.deepcopy(self.model.state_dict())
        global_model = ResNet18().to(self.device)
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
                prox_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    prox_term += (w - w_t).norm(2)
                loss += (self.args.mu / 2) * prox_term
                
                loss.backward()

                self.optimizer.step()   
                
        self.grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.model.named_parameters(),global_model.parameters())}
        self.parameters = {n:p.to(self.device) for n,p in self.model.named_parameters()}
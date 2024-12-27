from master import Master,Args
from client import fedavg

class NEKD_SGD(Master):
    def __init__(self,args : Args, random_init :bool = False , aggregation:str = "default"):
        super(NEKD_SGD,self).__init__(args,random_init,aggregation)

    
    def initialize_clients(self):
         self.clients = [fedavg(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.lr,self.weight_decay,self.opt_name,self.max_norm,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
         
from utils.utils_dataset import *
from utils.utils_libs import *
from utils.utils_eval import *
from utils.utils_metric import *
from utils.utils import *
from utils.utils_kd import *
from models import *
from client import *
import pandas as pd
import tqdm 
import json
from concurrent.futures import ThreadPoolExecutor
import copy


class Args:
    def __init__(self,dataset="CIFAR10",n_clients=10,training_rounds=50,learning_rate=1e-3,global_learning_rate=1,batch_size=100,weight_decay=1e-3,grad_aggregator=True,random_selection=False,
                 optimizer="sgd",loss_func="CELoss",rule="Drichlet",rule_arg=0.0,ub_sgm=0.0,aggr_scheme="FedAvg", local_epochs = 3,max_norm = 10,
                 beta = 1, beta1 = 0, beta2 = 0, alpha = 0, rho = 0, mu = 0, lambd =0 , gamma = 0, epsilon = 0, lr_decay = 0,
                 lambd1 = 1, lambd2 = 1, distill_epochs = 20, distill_lr = 1e-3, distill_batch_size = 256,
                 alpha1 = 0.5, alpha2 = 0.5,
                 leader_selection = "cyclic",
                 aggregation_mode = "default"
                 ):
        
        self.dataset = dataset
        self.n_clients = n_clients
        self.tr_rounds = training_rounds
        self.lr = learning_rate
        self.global_lr = global_learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = 42
        self.data = None
        self.opt_name = optimizer
        self.random_selection = random_selection if random_selection else self.n_clients
        self.loss_func = loss_func
        self.rule = "Drichlet"
        self.split_alpha = rule_arg
        self.local_epochs = local_epochs
        self.ub_sgm = ub_sgm
        self.method = aggr_scheme
        self.grad_aggregator = grad_aggregator
        self.max_norm = max_norm
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.rho = rho
        self.mu = mu
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd
        self.lr_decay = lr_decay
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.distill_epochs = distill_epochs
        self.distill_lr = distill_lr
        self.leader_selection = leader_selection
        self.aggregation_mode = aggregation_mode
        self.distill_batch_size = distill_batch_size
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        


class Master:
    def __init__(self,args : Args, random_init :bool = False , aggregation:str = "default"):
        
        self.dataset = args.dataset
        
        if self.dataset == "CIFAR10" or self.dataset == "MNIST":
            self.n_cls = 10
            self.global_model = ResNet18()
            
        self.n_clients = args.n_clients
        self.tr_rounds = args.tr_rounds
        self.lr = args.lr
        self.global_lr = args.global_lr
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.seed = 42
        self.data = None
        self.opt_name = args.opt_name
        self.random_init = random_init
        self.aggregation = aggregation
        self.random_selection = args.random_selection if args.random_selection else self.n_clients
        self.local_epochs = args.local_epochs
        self.lr_decay = args.lr_decay
        self.max_norm  = args.max_norm
        self.leader_selection = args.leader_selection
        if self.leader_selection == "cyclic":
            self.leader_cycle = np.arange(self.n_clients)
            np.random.shuffle(self.leader_cycle)
        elif self.leader_selection == "random":
            self.leader_cycle = np.array([random.randint(0, 9) for _ in range(10)])
        
        if args.loss_func == "CELoss":
            self.loss_func = torch.nn.CrossEntropyLoss()

        self.rule = args.rule
        if self.rule =="Drichlet":
            self.alpha = args.split_alpha
        
        self.aggregator = args.method
        self.ub_sgm = args.ub_sgm
        self.grad_aggregator = args.grad_aggregator
        self.args = args
        
        self.clients = []
        self.data_clients = []
        
        self.test = None
        self.global_grad = {name:torch.zeros_like(grad) for name,grad in self.global_model.named_parameters()}
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.metrics = {}
        self.correlations = {}
        self.r = 0
        self.reliability = torch.zeros((self.args.tr_rounds,self.n_clients))
        
        self.split_data()
        
    def run_experiment(self):
        
        self.initialize_clients()
        # Only applicable for MoFedSAM
        self.momentum =  param_to_vector(self.global_model)
        
        print("Data Distributed Among Clients")

        self.update_params()
        
        print("Model Parameters Initialized")

        self.train_and_eval()
        
        df = pd.DataFrame(self.reliability.numpy(), columns=[f'R_{i+1} ' for i in range(self.n_clients)])

        # Add an epoch index (optional)
        df.index.name = 'Epoch'
        df.to_csv("Results/"+self.data.name+"_"+self.aggregation+"_"+"_"+self.aggregator+"_"+self.opt_name+"_reliability.csv")
        
        self.results = pd.DataFrame.from_dict(self.metrics)
        
        self.results.to_csv("Results/"+self.data.name+"_"+self.aggregation+"_"+"_"+self.aggregator+"_"+self.opt_name+".csv")
        
    def split_data(self):
        self.data = DatasetObject(self.dataset,self.n_clients,self.seed,self.rule,self.ub_sgm,self.alpha)
        path = "Data/"+self.data.name
        self.trn_x = np.load(path+"/clnt_x.npy")
        self.trn_y = np.load(path+"/clnt_y.npy")
        self.test_x = np.load(path+"/tst_x.npy")
        self.test_y = np.load(path+"/tst_y.npy")

        self.data_loader = data.DataLoader(Dataset(self.test_x,self.test_y,dataset_name=self.dataset),self.batch_size,False)
        
        self.data_cardinality = np.array([self.trn_y[i].shape[0] for i in range(self.n_clients)])
        self.data_ratio = self.data_cardinality / np.sum(self.data_cardinality)
        self.test = Test(self.dataset,self.test_x,self.test_y,self.global_model,self.device)
    
    def initialize_clients(self):
         self.clients = [Client(self.dataset,self.trn_x[i],self.trn_y[i],self.batch_size,self.loss_func,self.opt_name,self.lr,self.weight_decay,10,self.device,self.grad_aggregator,self.args,self.local_epochs) for i in range(self.n_clients)]
    
    def update_params(self):
        for client in self.clients:
            client.update_parameters(self.global_model.parameters()) 
    
    def aggregate_grad(self):
        '''
        Uses gradient based aggregation scheme to compute global gadient
        If gradient based aggregator for parameter calculation, step the gradient to get global parameter
        '''
        self.old_global_model = copy.deepcopy(self.global_model)
        if self.args.aggregation_mode == "default":
            self.global_model = self.global_model.to(self.device)
            
            aggregated_gradients = {name:torch.zeros_like(grad).to(self.device) for name,grad in self.clients[0].grad.items()}
            
            for j,client in enumerate(self.clients):
                for i,(name,grad) in enumerate(client.grad.items()):
                    aggregated_gradients[name] += grad.to(self.device) * self.data_ratio[j]
            
            with torch.no_grad():
                for param, agg_grad in zip(self.global_model.parameters(), aggregated_gradients.values()):
                    param.grad = agg_grad.to(self.device)
            
            self.global_grad = {name:grad.to(self.device)*self.global_lr for name,grad in aggregated_gradients.items()}
            
            if self.grad_aggregator:
                with torch.no_grad():
                    for param, agg_grad in zip(self.global_model.parameters(), aggregated_gradients.values()):
                        param.data += self.global_lr * agg_grad
                        
            designated_leader = self.leader_cycle[self.r % self.n_clients]
            designated_leader = self.clients[designated_leader]
            teachers = [client.model for client in self.clients]
            student = self.global_model
            self.global_model,reliability,distill_loss = designated_leader.distill(self.args,teachers,student)
        
        
        elif self.args.aggregation_mode == "client":
            designated_leader_idx = self.leader_cycle[self.r % self.n_clients]
            designated_leader = self.clients[designated_leader_idx]
            teachers = [client.model for client in self.clients if client != designated_leader]
            student = ResNet18()
            student = student.to(self.device)
            student.load_state_dict(designated_leader.model.state_dict())
            student.train()
            new_student,reliability,distill_loss = designated_leader.distill(self.args,teachers,student)
            self.global_model.load_state_dict(new_student.state_dict())
            # Add zero for reliability at the designated leader idx
            reliability =  torch.cat((reliability[:, :designated_leader_idx], torch.zeros((1, 1)).to(self.device), reliability[:, designated_leader_idx:]), dim=1)
            reliability =  reliability.to(self.device)

        self.global_grad = { n:(l-g).to(self.device) for (n,l),g in zip(self.global_model.named_parameters(),self.old_global_model.parameters())}
        
        self.reliability[self.r] = reliability.unsqueeze(0)
        self.metrics[self.r] = {"Distillation Loss": distill_loss}
        
        
    
    def aggregate_params(self):
        '''
        Uses parameter-based aggregation schemes to get global parameters
        '''
        # if self.aggregation == "smart":
        #     self.smart_aggregation()

        if self.aggregation == "default":
            aggregated_params = [torch.zeros_like(param) for param in self.clients[0].model.parameters()]
            
            for j,client in enumerate(self.clients):
                    for i, grad in enumerate(client.model.parameters()):
                        aggregated_params[i] += grad * self.data_ratio[j]
                        
            with torch.no_grad():
                for param, aggregated_param in zip(self.global_model.parameters(),aggregated_params):
                    param.data.copy_(aggregated_param)
        
        
    
    def train(self):
        # for client in self.clients:
        #     client.train()
        random_select = random.sample(range(self.n_clients),self.random_selection)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(client.train) for i,client in enumerate(self.clients) if i in random_select]

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Client training failed: {e}")
        
    def test_metrics(self,round):
        acc,f1 =  self.test.test()
        if self.metrics.get(round) == None:
            self.metrics[round] = {}
        self.metrics[round]["Accuracy"] = acc
        self.metrics[round]["F1 Score"] = f1
    
    def gradient_eval(self,round):
        client_grads = [client.grad for client in self.clients]
        self.grad_metrics = Gradient_Metrics(self.n_clients,client_grads,self.global_grad)
        
        self.metrics[round] = self.grad_metrics.metrics
    
    def train_and_eval(self):
        for r in range(self.tr_rounds):
            # Global model communication and local training
            print("Training: Round",r+1,"/",self.tr_rounds)
            self.train()
            self.r = r
            print("Local Training Completed")
            #Global Gradient calculation (and update if relevant scheme) and related metric calculations
            self.aggregate_grad()
            self.gradient_eval(r)
            
            #Parametric aggregation (if not gradient based aggregation)
            if not self.grad_aggregator:
                self.aggregate_params()
            print("Global Aggregation Completed")

            # param_metrics = Parameter_Metrics(self.n_clients,[dict(client.model.named_parameters()) for client in self.clients],dict(self.global_model.named_parameters()))
            # self.metrics[r].update(param_metrics.metrics)
            # Send model update for next round
            self.update_params()
            
            #Record performance on test data (F1 and Accuracy)
            self.test_metrics(r)
            print("Training Round Complete!")
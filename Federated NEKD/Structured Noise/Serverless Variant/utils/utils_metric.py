from utils.utils_libs import *


class Gradient_Metrics:
    def __init__(self,n_clients,clients_gradients,global_grad):
        self.n = n_clients
        self.cln_grad = clients_gradients
        self.global_grad = global_grad

        self.metrics = {}
        
        self.calculate_metrics()
        
    def cosine_similarity(self,grad1, grad2):
        """
        Compute cosine similarity between two gradient tensors.
        
        Parameters:
        - grad1: First gradient tensor.
        - grad2: Second gradient tensor.
        
        Returns:
        - Cosine similarity value.
        """
        grad1_flat = torch.cat([g.flatten() for g in grad1])
        grad2_flat = torch.cat([g.flatten() for g in grad2])
        
        similarity = F.cosine_similarity(grad1_flat.to('cuda').unsqueeze(0), grad2_flat.to('cuda').unsqueeze(0))
        return similarity.item()
    
    def average_cosine_similarities(self):
        cln_sim = []
        global_sim = []
    
        # Compute cosine similarity for each pair of clients (client-client pairs similarities)
        for (cln_1_grad, cln_2_grad) in itertools.combinations(self.cln_grad, 2):
            sim = self.cosine_similarity(cln_1_grad.values(), cln_2_grad.values())
            cln_sim.append(sim)
        
        # Compute the average similarity
        avg_sim_cln = sum(cln_sim) / len(cln_sim) if cln_sim else 0
        
        for cln_i_grad in self.cln_grad:
            sim = self.cosine_similarity(cln_i_grad.values(),self.global_grad.values())
            global_sim.append(sim)
        
        avg_sim_global = sum(global_sim) / len(global_sim) if global_sim else 0
        
        return avg_sim_cln, avg_sim_global
    
    def average_variance(self):
        var = []
        for cln_i_grad in self.cln_grad:
            diff = [g1 - g2 for g1, g2 in zip(cln_i_grad.values(), self.global_grad.values())]
            diff_flat = torch.cat([d.flatten() for d in diff])
            var.append((diff_flat).norm(2).detach().cpu()) 

        return np.mean(np.array(var))
    
    def avg_diff_norms(self):
        diff_norm = []
        for (cln_1_grad, cln_2_grad) in itertools.combinations(self.cln_grad, 2):
            norm_1 = torch.linalg.vector_norm(torch.cat([g.flatten() for g in cln_1_grad.values()]),1)
            norm_2 = torch.linalg.vector_norm(torch.cat([g.flatten() for g in cln_2_grad.values()]),1)

            diff_norm.append(torch.abs((norm_1 - norm_2)).detach().cpu())

        return np.mean(np.array(diff_norm))
        
    def calculate_metrics(self):
        
        # self.metrics["Average Cosine Similarity (Pairwise Grad)"],self.metrics["Average Cosine Similarity (Global Grad)"]  = self.average_cosine_similarities()
        self.metrics["Variance Grad"] = self.average_variance()
        # self.metrics["Difference of Norm (Grad)"] = self.avg_diff_norms()
        
        
class Hessian_Metrics:
    def __init__(self,n_clients,client_hessians):
        self.n = n_clients
        self.cln_hessian = client_hessians
        self.metrics = {}
        
        self.calculate_metrics()
    
    def calculate_metrics(self):
        
        for name in self.cln_hessian.keys():
            dets = torch.tensor([torch.linalg.det(hessian[name]) for hessian in self.cln_hessian]).flatten()
            fb_norm = torch.tensor([torch.linalg.matrix_norm(hessian[name]) for hessian in self.cln_hessian]).flatten()
            nuclear_norm = torch.tensor([torch.norm(hessian[name],'nuc') for hessian in self.cln_hessian]).flatten()
            trace = torch.tensor([torch.trace(hessian[name]) for hessian in self.cln_hessian]).flatten()
            cond = torch.tensor([torch.linalg.cond(hessian[name]) for hessian in self.cln_hessian]).flatten()
            singulars = [torch.linalg.svdvals(hessian[name]) for hessian in self.cln_hessian]

            self.metrics[name] = {}
            self.metrics[name]["Det."] = dets
            self.metrics[name]["Fb. Norm"] = fb_norm
            self.metrics[name]["Nuc. Norm"] = nuclear_norm
            self.metrics[name]["Trace"] = trace
            self.metrics[name]["Condition Number"] = cond
            self.metrics[name]["Singulars"] = singulars
             
class Parameter_Metrics:
    def __init__(self,n_clients,clients_params,global_params):
        self.n = n_clients
        self.cln_params = clients_params
        self.global_params = global_params

        self.metrics = {}
        
        self.calculate_metrics()
        
    def cosine_similarity(self,p1, p2):
        """
        Compute cosine similarity between two parameter tensors.
        
        Parameters:
        - p1: First parameter tensor.
        - p2: Second parameter tensor.
        
        Returns:
        - Cosine similarity value.
        """
        p1_flat = torch.cat([g.flatten() for g in p1])
        p2_flat = torch.cat([g.flatten() for g in p2])
        
        similarity = F.cosine_similarity(p1_flat.unsqueeze(0), p2_flat.unsqueeze(0))
        return similarity.item()
    
    def average_cosine_similarities(self):
        cln_sim = []
        global_sim = []
    
        # Compute cosine similarity for each pair of clients (client-client pairs similarities)
        for (cln_1_p, cln_2_p) in itertools.combinations(self.cln_params, 2):
            sim = self.cosine_similarity(cln_1_p.values(), cln_2_p.values())
            cln_sim.append(sim)
        
        # Compute the average similarity
        avg_sim_cln = sum(cln_sim) / len(cln_sim) if cln_sim else 0
        
        for cln_i_p in self.cln_params:
            sim = self.cosine_similarity(cln_i_p.values(),self.global_params.values())
            global_sim.append(sim)
        
        avg_sim_global = sum(global_sim) / len(global_sim) if global_sim else 0
        
        return avg_sim_cln, avg_sim_global
    
    def average_variance(self):
        var = []
        for cln_i_param in self.cln_params:
            diff = [g1 - g2 for g1, g2 in zip(cln_i_param.values(), self.global_params.values())]
            diff_flat = torch.cat([d.flatten() for d in diff])
            var.append((diff_flat).norm(2).detach().cpu()) 

        return np.mean(np.array(var))

    def avg_diff_norms(self):
        diff_norm = []
        for (cln_1_param, cln_2_param) in itertools.combinations(self.cln_params, 2):
            norm_1 = torch.linalg.vector_norm(torch.cat([g.flatten() for g in cln_1_param.values()]),1)
            norm_2 = torch.linalg.vector_norm(torch.cat([g.flatten() for g in cln_2_param.values()]),1)

            diff_norm.append(torch.abs((norm_1 - norm_2)).detach().cpu())

        return np.mean(np.array(diff_norm))
        
    def calculate_metrics(self):
        
        self.metrics["Average Cosine Similarity (Pairwise Param)"],self.metrics["Average Cosine Similarity (Global Param)"]  = self.average_cosine_similarities()
        self.metrics["Variance Param"] = self.average_variance()
        self.metrics["Difference of Norm (Param)"] = self.avg_diff_norms()
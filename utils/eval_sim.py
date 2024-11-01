import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance, spearmanr, ks_2samp
from scipy.special import rel_entr

def ks_test(batch_p, batch_q):
    """
    Computes the average two-sample KS test statistic across a batch.
    
    Parameters:
        batch_p (torch.Tensor): Batch of teacher logits (batch_size, num_classes).
        batch_q (torch.Tensor): Batch of student logits (batch_size, num_classes).
    
    Returns:
        float: Average KS statistic across the batch.
    """
    ks_stats = []
    for i in range(batch_p.size(0)):
        ks_stat, _ = ks_2samp(batch_p[i].cpu().numpy(), batch_q[i].cpu().numpy())
        ks_stats.append(ks_stat)
    return torch.tensor(ks_stats).mean().item()

def kl_divergence(batch_p, batch_q):
    """
    Computes the average KL divergence across a batch.
    
    Parameters:
        batch_p (torch.Tensor): Teacher logits (batch_size, num_classes).
        batch_q (torch.Tensor): Student logits (batch_size, num_classes).
    
    Returns:
        float: Average KL divergence across the batch.
    """
    p_log_softmax = F.log_softmax(batch_p, dim=-1)
    q_softmax = F.softmax(batch_q, dim=-1)
    return F.kl_div(p_log_softmax, q_softmax, reduction='batchmean').item()

def spearman_rank_corr(batch_p, batch_q):
    """
    Computes the average Spearman rank correlation coefficient across a batch.
    
    Parameters:
        batch_p (torch.Tensor): Teacher logits (batch_size, num_classes).
        batch_q (torch.Tensor): Student logits (batch_size, num_classes).
    
    Returns:
        float: The average Spearman rank correlation across the batch.
    """
    rank_correlations = []
    for i in range(batch_p.size(0)):
        rank_p = torch.argsort(torch.argsort(batch_p[i]))
        rank_q = torch.argsort(torch.argsort(batch_q[i]))
        corr, _ = spearmanr(rank_p.cpu().numpy(), rank_q.cpu().numpy())
        rank_correlations.append(corr)
    return torch.tensor(rank_correlations).mean().item()

def wasserstein_distance_batch(batch_p, batch_q):
    """
    Computes the average Wasserstein distance across a batch.
    
    Parameters:
        batch_p (torch.Tensor): Teacher logits (batch_size, num_classes).
        batch_q (torch.Tensor): Student logits (batch_size, num_classes).
    
    Returns:
        float: Average Wasserstein distance across the batch.
    """
    wasserstein_dists = []
    for i in range(batch_p.size(0)):
        dist = wasserstein_distance(batch_p[i].cpu().numpy(), batch_q[i].cpu().numpy())
        wasserstein_dists.append(dist)
    return torch.tensor(wasserstein_dists).mean().item()

def evaluate_similarities(teacher, student, test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    student = student.to(device)
    teacher = teacher.to(device)
    sim_kl = 0
    sim_wassertein = 0 
    sim_dks = 0
    sim_src = 0
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        student_feats,_ = student(batch_x)
        teacher_feats,_ = teacher(batch_x)
        student_probs = F.softmax(student_feats,dim=-1) 
        teacher_probs = F.softmax(teacher_feats,dim=-1)
        sim_kl += kl_divergence(teacher_probs,student_probs)
        sim_wassertein += wasserstein_distance_batch(teacher_probs,student_probs)
        sim_dks += ks_test(teacher_probs,student_probs)
        sim_src += spearman_rank_corr(teacher_probs,student_probs)
    
    n = len(test_loader)
    return sim_kl/n , sim_dks/n , sim_wassertein/n , sim_src/n 





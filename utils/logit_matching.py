import torch

def basic_logic_matching(student_model,
                         teacher_model,
                         batch_x, batch_y,
                         optimizer,device,temperature=2,weight=0.75):
    '''
    Carries out one iteration of logit matching with temperature on a single batch of data
    Follows Hinton's paper
    '''
    loss_func = torch.nn.CrossEntropyLoss()
    for params in teacher_model.parameters():
        params.requires_grad = False
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    logits_s = student_model(batch_x)
    logits_t = teacher_model(batch_x)
    log_soft_s = torch.nn.functional.log_softmax(logits_s/temperature,dim=-1) 
    soft_t = torch.nn.functional.softmax(logits_t/temperature,dim=-1)
    loss_dist =  torch.sum(soft_t * (soft_t.log() - log_soft_s)) / log_soft_s.size()[0] * (temperature**2)
    loss_ce = loss_func(logits_s,batch_y.reshape(-1).long())
    loss = loss_ce *(1-weight) + weight * loss_dist
    pred_list = torch.argmax(logits_s,dim=1) == batch_y
    loss.backward()
    optimizer.step()

    return loss.item(),pred_list

def lsr_logit_matching(student_model,
                         teacher_model,
                         batch_x, batch_y,
                         optimizer,device,epsilon=0.1,weight=0.75):
    '''
    Carries out one iteration of logit matching on a single batch of data
    Follows LSR with the smoothing teacher logits as: (1-epsilon) delta_{k,y} l(k) + epsilon/N + l(k) (1-delta{k,y})
    in addition to label smoothing: (1-epsilon) delta_{k,y}+ epsilon/N
    Here N is the number of classes i.e. assumes uniform distribution
    delta{k,y} indicates removing a mass only from the true class (is 1 only when k = y)
    l(k) is the logit value for the kth class 
    '''
    loss_func = torch.nn.CrossEntropyLoss(reduction='none',label_smoothing=epsilon) # torch has in-built support for label smoothing
    # We only extend and code the logit smoothing for the teacher model
    for params in teacher_model.parameters():
        params.requires_grad = False
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    logits_s = student_model(batch_x)
    logits_t = teacher_model(batch_x)
    n_classes = logits_t.size(-1)
    max_class_t = torch.argmax(logits_t, dim=-1, keepdim=True) # find the index of the peak in the logits
    one_hot_max_t = torch.zeros_like(logits_t).scatter_(1, max_class_t, 1).to(device)  # Make it a one hot tensor per row
    # Smooth the logits
    smooth_logits_t = logits_t - one_hot_max_t * epsilon * logits_t  # Reduce mass from the peak
    smooth_logits_t = smooth_logits_t + epsilon / n_classes  # Redistribute the mass evenly to others
    
    # The rest of the routine continues as normal with temperature being effectively replaced with LSR (only L is now logits)
    log_s = torch.nn.functional.log_softmax(logits_s,dim=-1) 
    loss_dist =  torch.sum(smooth_logits_t * (smooth_logits_t.log() - log_s)) / log_s.size()[0]
    loss_ce = loss_func(logits_s,batch_y.reshape(-1).long())
    loss = loss_ce *(1-weight) + weight * loss_dist
    pred_list = torch.argmax(logits_s,dim=1) == batch_y
    loss.backward()
    optimizer.step()

    return loss.item(),pred_list


def decoupled_logit_matching(student_model,
                         teacher_model,
                         batch_x, batch_y,
                         optimizer,device, temperature = 2,
                         weight=0.75,alpha=1.0,beta=8.0):
    '''
    Carries out one iteration of logit matching on a single batch of data
    Follows Decoupled Knowledge Distillation where the distillation of the gold class and the other classes is decoupled 
    This is done by two loss functions (two KL divergences) which are weighted as:
    alpha * TCKD + beta * NCKD
    in addition to the original weighting between CE and Distillation Loss
    '''
    loss_func = torch.nn.CrossEntropyLoss()
    for params in teacher_model.parameters():
        params.requires_grad = False
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    logits_s = student_model(batch_x)
    logits_t = teacher_model(batch_x)
    
    
    gold_mask = torch.zeros_like(logits_s).scatter_(1,batch_y.reshape(-1),1)
    other_mask = 1 - gold_mask
    gold_mask = gold_mask.bool()
    other_mask = other_mask.bool()
    
    soft_s = torch.nn.functional.softmax(logits_s/temperature,dim=-1)
    soft_t = torch.nn.functional.softmax(logits_t/temperature,dim=-1)
    
    gold_masked_soft_s = (gold_mask*soft_s).sum(dim=-1,keepdim=True)
    other_masked_soft_s = (other_mask*soft_t).sum(dim=-1,keepdim=True)
    
    gold_masked_soft_t = (gold_mask*soft_t).sum(dim=-1,keepdim=True)
    other_masked_soft_t = (other_mask*soft_t).sum(dim=-1,keepdim=True)
    
    cat_soft_s = torch.cat([gold_masked_soft_s,other_masked_soft_s],dim=-1)
    cat_soft_t = torch.cat([gold_masked_soft_t,other_masked_soft_t],dim=-1)
    
    log_cat_soft_s = torch.log(cat_soft_s)
    
    tckd_loss_dist =  torch.nn.functional.kl_div(log_cat_soft_s,cat_soft_t,size_average=False)/log_cat_soft_s.size()[0] * (temperature**2)
    
    log_soft_s_nc = torch.nn.functional.log_softmax(logits_s/temperature - gold_mask * 10000.0, dim=-1) 
    soft_t_nc = torch.nn.functional.softmax(logits_t/temperature - gold_mask * 10000.0, dim=-1) 
    
    nckd_loss_dist = torch.nn.functional.kl_div(log_soft_s_nc, soft_t_nc, size_average=False)* (temperature**2)/ batch_y.size()[0]
    
    loss_dist = alpha * tckd_loss_dist + beta * nckd_loss_dist
    loss_ce = loss_func(logits_s,batch_y.reshape(-1).long())
    loss = loss_ce *(1-weight) + weight * loss_dist
    pred_list = torch.argmax(logits_s,dim=1) == batch_y
    loss.backward()
    optimizer.step()

    return loss.item(),pred_list


def independent_train(model,train_loader,loss_func,optimizer,device):
    model = model.to(device)
    model.train()
    total_loss = 0
    pred_list = None
    
    for i,batch in enumerate(train_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = loss_func(pred,y_batch.reshape(-1).long())
        total_loss += loss.item()
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch
        loss.backward()
        optimizer.step()

    return total_loss/len(train_loader),pred_list.float().mean().cpu().numpy()
    

def eval(model,test_loader,loss_func,device):
    total_loss = 0
    for i,batch in enumerate(test_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)

        loss = loss_func(pred,y_batch.reshape(-1).long())
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

        total_loss += loss.item()

    total_loss /= len(test_loader)

    return total_loss,pred_list.float().mean().cpu().numpy()

def train(student_model,teacher_model,train_loader,optimizer,device,types="basic"):
    '''
    Carries out one epoch of training through distillation
    '''
    total_loss = 0
    pred_list = None
    for params in teacher_model.parameters():
        params.requires_grad = False
    for i,batch in enumerate(train_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        if types == "basic":
            loss,preds = basic_logic_matching(student_model,teacher_model,x_batch,y_batch,optimizer,device)
        elif types == "lsr":
            loss,preds = lsr_logit_matching(student_model,teacher_model,x_batch,y_batch,optimizer,device)
        elif types == "dkd":
            loss,preds = decoupled_logit_matching(student_model,teacher_model,x_batch,y_batch,optimizer,device)
        
        total_loss += loss
        pred_list = torch.cat([pred_list,preds]) if pred_list is not None else preds
    
    total_loss /= len(train_loader)

    return total_loss,pred_list.float().mean().cpu().numpy()
        
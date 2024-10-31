import torch

def basic_logic_matching(student_model,
                         teacher_model,
                         batch_x, batch_y,
                         optimizer,device,temperature=2,weight=0.75,f=True):
    '''
    Carries out one iteration of logit matching with temperature on a single batch of data
    Follows Hinton's paper
    '''
    loss_func = torch.nn.CrossEntropyLoss()
    for params in teacher_model.parameters():
        params.requires_grad = False
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    if not f:
        logits_s = student_model(batch_x)
        logits_t = teacher_model(batch_x)
    else:
        logits_s,_ = student_model(batch_x) 
        logits_t,_ = teacher_model(batch_x)
    log_soft_s = torch.nn.functional.log_softmax(logits_s/temperature,dim=-1) 
    soft_t = torch.nn.functional.softmax(logits_t/temperature,dim=-1)
    loss_dist =  torch.sum(soft_t * (soft_t.log() - log_soft_s)) / log_soft_s.size()[0] * (temperature**2)
    loss_ce = loss_func(logits_s,batch_y.reshape(-1).long())
    loss = loss_ce *(1-weight) + weight * loss_dist
    pred_list = torch.argmax(logits_s,dim=1) == batch_y
    loss.backward()
    optimizer.step()

    return loss.item(),pred_list

def lsr_logit_matching(student_model, teacher_model, batch_x, batch_y, optimizer, device, epsilon=0.05, weight=0.5,f=True):
    '''
    Carries out one iteration of logit matching on a single batch of data.
    Follows LSR with the smoothing teacher logits.
    '''
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=epsilon)

    # Freeze teacher model parameters
    for params in teacher_model.parameters():
        params.requires_grad = False

    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()

    if not f:
        logits_s = student_model(batch_x)
        logits_t = teacher_model(batch_x)
    else:
        logits_s,_ = student_model(batch_x) 
        logits_t,_ = teacher_model(batch_x)

    logits_s += 1e-16
    logits_t += 1e-16
    n_classes = logits_t.size(-1)

    # Calculate softmax probabilities for teacher and student logits
    probs_t = torch.nn.functional.softmax(logits_t, dim=-1)
    probs_s = torch.nn.functional.softmax(logits_s, dim=-1)

    # Smooth the teacher probabilities
    smooth_probs_t = (1 - epsilon) * probs_t + epsilon / n_classes
    # Smooth the student probabilities
    smooth_probs_s = (1 - epsilon) * probs_s + epsilon / n_classes

    # Calculate distillation loss using KL divergence with `kl_div` function for stability
    loss_dist = torch.nn.functional.kl_div(smooth_probs_s.log(), smooth_probs_t, reduction='batchmean')

    # Calculate CE loss
    loss_ce = loss_func(logits_s, batch_y.reshape(-1).long())
    
    # Combine losses
    loss = loss_ce * weight + loss_dist

    # Calculate predictions and backpropagate
    loss.backward()
    optimizer.step()
    pred_list = torch.argmax(logits_s, dim=1) == batch_y

    return loss.item(), pred_list


# def lsr_logit_matching(student_model,
#                        teacher_model,
#                        batch_x, batch_y,
#                        optimizer, device, epsilon=0.05, weight=0.5):
#     '''
#     Carries out one iteration of logit matching on a single batch of data.
#     Follows LSR with the smoothing teacher logits.
#     '''
#     loss_func = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=epsilon)

#     # Freeze teacher model parameters
#     for params in teacher_model.parameters():
#         params.requires_grad = False

#     batch_x, batch_y = batch_x.to(device), batch_y.to(device)
#     optimizer.zero_grad()

#     logits_s = student_model(batch_x) + 1e-16
#     logits_t = teacher_model(batch_x) + 1e-16
#     n_classes = logits_t.size(-1)

#     # Identify the peak logit and create a one-hot tensor
#     max_class_t = torch.argmax(logits_t, dim=-1, keepdim=True)
#     one_hot_max_t = torch.zeros_like(logits_t).scatter_(1, max_class_t, 1).to(device)

#     max_class_s = torch.argmax(logits_s, dim=-1, keepdim=True)
#     one_hot_max_s = torch.zeros_like(logits_s).scatter_(1, max_class_s, 1).to(device)

#     probs_t = torch.nn.functional.softmax(logits_t, dim=-1)
#     # Smooth the teacher logits
#     smooth_probs_t = torch.max(probs_t - one_hot_max_t * epsilon * probs_t,torch.tensor([1e-16]).to(device))  # Reduce mass from peak
#     smooth_probs_t = smooth_probs_t + epsilon *torch.max(logits_t,dim=-1,keepdim=True)[0]/ n_classes  # Redistribute mass evenly to others

#     # Smooth the student logits
#     probs_s = torch.nn.functional.softmax(logits_s, dim=-1)
#     smooth_probs_s = torch.max(probs_s - one_hot_max_s * epsilon * probs_s,torch.tensor([1e-16]).to(device))  # Reduce mass from peak
#     smooth_probs_s = smooth_probs_s + epsilon *torch.max(logits_t,dim=-1,keepdim=True)[0]/ n_classes  # Redistribute mass evenly to others

#     # Calculate distillation loss using KL divergence

#     loss_dist =  torch.sum(smooth_probs_t * (smooth_probs_t.log() - smooth_probs_s.log())) / smooth_probs_s.size()[0]
#     # Calculate CE loss
#     loss_ce = loss_func(logits_s, batch_y.reshape(-1).long())
#     # Combine losses
#     loss = loss_ce * weight + loss_dist

#     # Calculate predictions and backpropagate
#     loss.backward()
#     optimizer.step()
#     pred_list = torch.argmax(logits_s, dim=1) == batch_y

#     return loss.item(), pred_list



def decoupled_logit_matching(student_model,
                             teacher_model,
                             batch_x, batch_y,
                             optimizer, device, temperature=2,
                             weight=0.75, alpha=1.0, beta=1.0, f= True):
    '''
    Carries out one iteration of logit matching on a single batch of data.
    '''
    loss_func = torch.nn.CrossEntropyLoss()

    # Freeze teacher model parameters
    for params in teacher_model.parameters():
        params.requires_grad = False

    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()

    if not f:
        logits_s = student_model(batch_x)
        logits_t = teacher_model(batch_x)
    else:
        logits_s,_ = student_model(batch_x) 
        logits_t,_ = teacher_model(batch_x)

    # Create gold and other class masks
    gold_mask = torch.zeros_like(logits_s).scatter_(1, batch_y.reshape(-1).unsqueeze(-1), 1)
    other_mask = torch.ones_like(logits_s).scatter_(1, batch_y.reshape(-1).unsqueeze(-1), 0)
    gold_mask = gold_mask.bool()
    other_mask = other_mask.bool()

    # Compute softened logits with temperature scaling
    soft_s = torch.nn.functional.softmax(logits_s / temperature, dim=-1) + 1e-16
    soft_t = torch.nn.functional.softmax(logits_t / temperature, dim=-1) + 1e-16

    # Separate gold and non-gold probabilities for student and teacher
    gold_masked_soft_s = (gold_mask * soft_s).sum(dim=-1, keepdim=True)
    other_masked_soft_s = (other_mask * soft_s).sum(dim=-1, keepdim=True)

    gold_masked_soft_t = (gold_mask * soft_t).sum(dim=-1, keepdim=True)
    other_masked_soft_t = (other_mask * soft_t).sum(dim=-1, keepdim=True)

    # Concatenate gold and non-gold probabilities
    cat_soft_s = torch.cat([gold_masked_soft_s, other_masked_soft_s], dim=-1)
    cat_soft_t = torch.cat([gold_masked_soft_t, other_masked_soft_t], dim=-1)

    # Compute TCKD Loss (Gold Class KL Divergence)
    log_cat_soft_s = torch.log(cat_soft_s)
    tckd_loss_dist = torch.nn.functional.kl_div(log_cat_soft_s, cat_soft_t, reduction='batchmean') * (temperature ** 2)

    # Mask out gold class for NCKD (Non-Gold Class KL Divergence)
    log_soft_s_nc = torch.nn.functional.log_softmax(logits_s / temperature - gold_mask * 1000000.0, dim=-1)
    soft_t_nc = torch.nn.functional.softmax(
        logits_t / temperature - gold_mask * 1000000.0, dim=-1
    )

    nckd_loss_dist = torch.nn.functional.kl_div(log_soft_s_nc, soft_t_nc, reduction='batchmean') * (temperature ** 2)

    # Combine TCKD and NCKD losses
    loss_dist = alpha * tckd_loss_dist + beta * nckd_loss_dist

    # Compute standard CE loss
    loss_ce = loss_func(logits_s, batch_y.long())
    loss = loss_ce * (1-weight) + weight* loss_dist

    # Calculate predictions and backpropagate
    loss.backward()
    optimizer.step()
    pred_list = torch.argmax(logits_s, dim=1) == batch_y

    return loss.item(), pred_list


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
    pred_list = None
    for i,batch in enumerate(test_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)

        loss = loss_func(pred,y_batch.reshape(-1).long())
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

        total_loss += loss.item()

    total_loss /= len(test_loader)

    return total_loss,pred_list.float().mean().cpu().numpy()

def train(student_model,teacher_model,train_loader,optimizer,device,types="basic",f=True):
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
            loss,preds = basic_logic_matching(student_model,teacher_model,x_batch,y_batch,optimizer,device,f=f)
        elif types == "lsr":
            loss,preds = lsr_logit_matching(student_model,teacher_model,x_batch,y_batch,optimizer,device,f=f)
        elif types == "dkd":
            loss,preds = decoupled_logit_matching(student_model,teacher_model,x_batch,y_batch,optimizer,device,f=f)

        total_loss += loss
        pred_list = torch.cat([pred_list,preds]) if pred_list is not None else preds

    total_loss /= len(train_loader)

    return total_loss,pred_list.float().mean().cpu().numpy()

def eval_f(model,test_loader,loss_func,device):
    total_loss = 0
    pred_list = None
    for i,batch in enumerate(test_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred,_ = model(x_batch)
        loss = loss_func(pred,y_batch.reshape(-1).long())
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch

        total_loss += loss.item()

    total_loss /= len(test_loader)

    return total_loss,pred_list.float().mean().cpu().numpy()

def independent_train_f(model,train_loader,loss_func,optimizer,device):
    model = model.to(device)
    model.train()
    total_loss = 0
    pred_list = None

    for i,batch in enumerate(train_loader):
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred,_ = model(x_batch)
        loss = loss_func(pred,y_batch.reshape(-1).long())
        total_loss += loss.item()
        pred_list = torch.cat([pred_list,torch.argmax(pred,dim=1) == y_batch]) if pred_list is not None else torch.argmax(pred,dim=1) == y_batch
        loss.backward()
        optimizer.step()

    return total_loss/len(train_loader),pred_list.float().mean().cpu().numpy()

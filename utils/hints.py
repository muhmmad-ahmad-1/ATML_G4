import torch
import torch.nn as nn
import torch.functional as F

def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes

class ConvReg(nn.Module):
    """Convolutional regression to match feature map sizes (student to teacher)"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

def regressor_block(teacher,student,feature_idx):
    s_shape = get_feat_shapes(student,teacher,(32,32))[0][feature_idx]
    t_shape = get_feat_shapes(student,teacher,(32,32))[1][feature_idx]
    return ConvReg(s_shape,t_shape,use_relu=True)

class FitNet:
    def __init__(self,teacher,student,feature_idx,weight=0.75):
        self.regressor = regressor_block(teacher,student,feature_idx)
        self.teacher = teacher
        self.student = student
        self.feature_idx = feature_idx
        self.weight = weight

    
    def forward(self,batch_x,batch_y,train=True):
        with torch.no_grad():
            _,teacher_feats = self.teacher(batch_x)
            teacher_feats = teacher_feats["feats"][self.feature_idx]
        logits_student,student_feats = self.student(batch_x)
        student_feats = student_feats["feats"][self.feature_idx]
        scaled_student_feats = self.regressor(student_feats)

        loss_ce = nn.CrossEntropyLoss()(logits_student, batch_y.reshape(-1).long())
        if train:
            loss_distill = nn.MSELoss()(scaled_student_feats,teacher_feats)
            loss = (1-self.weight) * loss_ce + self.weight * loss_distill
            
        else:
            loss = loss_ce

        return logits_student,loss

    def get_learnable_params(self):
        return list(self.student.parameters()) + list(self.regressor.parameters())

def train_hints(model,train_loader,optimizer,device):
    model.student.train()
    model.teacher.eval()
    model.regressor.eval()
    pred_list = None
    total_loss = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_x,batch_y = batch_x.to(device),batch_y.to(device) 
        logits,loss = model.forward(batch_x,batch_y)
        loss.backward()
        optimizer.step()
        pred_list = torch.cat([pred_list,torch.argmax(logits,dim=1) == batch_y]) if pred_list is not None else torch.argmax(logits,dim=1) == batch_y
        total_loss += loss.item()
    
    return total_loss/len(train_loader),pred_list.float().mean().cpu().numpy()

def eval_hints(model,test_loader,device):
    model.student.eval()
    model.teacher.eval()
    model.regressor.eval()
    pred_list = None
    total_loss = 0
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x,batch_y = batch_x.to(device),batch_y.to(device) 
        logits,loss = model.forward(batch_x,batch_y,train=False)
        pred_list = torch.cat([pred_list,torch.argmax(logits,dim=1) == batch_y]) if pred_list is not None else torch.argmax(logits,dim=1) == batch_y
        total_loss += loss.item()
    
    return total_loss/len(test_loader),pred_list.float().mean().cpu().numpy()
            
        
def create_model(teacher,student,feature_idx,weight):
    model = FitNet(teacher,student,feature_idx,weight)
    return model


import torch 
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchsummary import summary
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''DISCRIMINATIVE MODELS''''''''''''''''''''''''''''''''''''''''''''''''''

class ResNet18(nn.Module):
    def __init__(self,n_classes,freeze=True):
        super().__init__()
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        
        if n_classes != 1000:
            self.model.fc = nn.Linear(512,n_classes)
        
        if freeze:
            for name,params in self.model.named_parameters():
                if name in ['fc.weight','fc.bias']:
                    continue
                params.requires_grad = False
    
    def forward(self,x):
        x = self.model(x)
        return x

class ViTBase(nn.Module):
    def __init__(self,n_classes,freeze=True):
        super().__init__()
        
        if n_classes != 1000:
            self.model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=n_classes)
        
        else:
            self.model = timm.create_model('vit_base_patch16_224',pretrained=True)
        
        if freeze:
            for params in self.model.parameters():
                params.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True
            for name,params in self.model.named_parameters():
                print(name,params.requires_grad)
                
    def forward(self,x):
        x = self.model(x)
        return x

class ViTSmall(nn.Module):
    def __init__(self,n_classes,freeze=True):
        super().__init__()
        
        if n_classes != 1000:
            self.model = timm.create_model('vit_small_patch16_224',pretrained=True,num_classes=n_classes)
        
        else:
            self.model = timm.create_model('vit_small_patch16_224',pretrained=True)
        
        if freeze:
            for params in self.model.parameters():
                params.requires_grad = False
            for param in self.model.head.parameters():
                param.requires_grad = True
            for name,params in self.model.named_parameters():
                print(name,params.requires_grad)
                
    def forward(self,x):
        x = self.model(x)
        return x

#'''''''''''''''''''''''''''''''''''''''''''''''CONTRASTIVE MODEL'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class ImageEncoder(nn.Module):
    def __init__(self,emb_dim,freeze=True):
        super().__init__()
        self.image_encoder = timm.create_model('vit_base_patch32_clip_224',pretrained=True, num_classes = 0)
        
        for params in self.image_encoder.parameters():
            params.requires_grad = False
        
        if emb_dim != 768:
            self.img_proj = nn.Linear(768,emb_dim)
        else:
            self.img_proj = None
    
    def forward(self,x):
        if self.img_proj is None:
            return self.image_encoder(x)
        else:
            return self.img_proj(self.image_encoder(x))
    
class TextEncoder(nn.Module):
    def __init__(self, 
                 emb_dim,
                 freeze = True):
        super(TextEncoder,self).__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased',config=DistilBertConfig())
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        if emb_dim != 768:
            self.projection = nn.Linear(768,emb_dim)
        else:
            self.projection = None
            
    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor):
        out = self.model(input_ids=input_ids,attention_mask=attention_mask)
        if self.projection is None:
            return out[0][:,0]
        else:
            return self.projection(out[0][:,0])

class CLIPModel(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        
        self.image_encoder = ImageEncoder(emb_dim=emb_dim)
        self.text_encoder = TextEncoder(emb_dim=emb_dim)
    
    def forward(self,
                x: dict):
        
        image = x['image']
        text = x['caption']
        input_ids = text['input_ids']
        attn_mask = text['attention_mask']
        
        image_emb = self.image_encoder(image)
        
        text_emb = self.text_encoder(input_ids=input_ids,attention_mask=attn_mask)
        
        logits = image_emb @ text_emb.T
        
        return logits

if __name__ == "__main__":
    model = ImageEncoder(10)

    
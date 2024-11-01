import torch
import torch.nn as nn
from torchvision.models import VGG11_Weights,VGG16_Weights,VGG19_Weights, VGG11_BN_Weights, VGG16_BN_Weights, VGG19_BN_Weights
from torchvision.models import vgg11,vgg16,vgg19

class VGG11(nn.Module):
    def __init__(self,n_classes,teacher=False,device="cuda"):
        super(VGG11,self).__init__()
        if teacher:
            self.model = vgg11(weights=VGG11_Weights.DEFAULT,progress=True)
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            self.model = vgg11()
        if n_classes != 1000:
            self.model.classifier[3].requires_grad = True
            self.model.classifier[6] = nn.Linear(4096,n_classes,device=device)
    
    def forward(self,x):
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self,n_classes,teacher=True,device="cuda"):
        super(VGG16,self).__init__()
        if teacher:
            self.model = vgg16(weights=VGG16_Weights.DEFAULT,progress=True)
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            self.model = vgg16()
        if n_classes != 1000:
            self.model.classifier[3].requires_grad = True
            self.model.classifier[6] = nn.Linear(4096,n_classes,device=device)
    
    def forward(self,x):
        return self.model(x)

class VGG19(nn.Module):
    def __init__(self,n_classes,teacher=False,device="cuda"):
        super(VGG19,self).__init__()
        if teacher:
            self.model = vgg19(weights=VGG19_Weights.DEFAULT,progress=True)
            for params in self.model.parameters():
                params.requires_grad = False
        else:
            self.model = vgg19()
        if n_classes != 1000:
            self.model.classifier[3].requires_grad = True
            self.model.classifier[6] = nn.Linear(4096,n_classes,device=device)
    
    def forward(self,x):
        return self.model(x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# __all__ = [
#     "VGG",
#     "vgg11",
#     "vgg11_bn",
#     "vgg16",
#     "vgg16_bn",
#     "vgg19_bn",
#     "vgg19",
# ]

class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential( nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=100, bias=True))
        
        self._initialize_weights()
        self.stage_channels = [c[-1] for c in cfg]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return self.stage_channels

    def forward(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x
        x = self.pool4(x)
        x = x.reshape(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3, f4]
        feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre, f4_pre]
        feats["pooled_feat"] = f5

        return x, feats

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


cfg = {
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [
        [64, 64],
        [128, 128],
        [256, 256, 256, 256],
        [512, 512, 512, 512],
        [512, 512, 512, 512],
    ],
}

def vgg11_f(pretrained=True,n_classes=100,ref_dict=None):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["A"])
    if pretrained:
        print("Loading pretrained weights")
        # Load pretrained weights from torchvision
        weights = ref_dict
        # Load pretrained VGG16
        dict = model.state_dict()

        # Mapping logic: Directly map "features.*" layers to corresponding "block*.*" layers
        pretrained_keys = [k for k in weights.keys() if "features" in k]
        vgg_index = 0

        for block_name in ["block0", "block1", "block2", "block3", "block4"]:
            # Get the number of layers in each block from the custom model's configuration
            num_layers_in_block = len([k for k in dict.keys() if block_name in k])
            for layer_idx in range(num_layers_in_block):
                # Map VGG16 layer weights to corresponding custom model layer
                custom_weight_key = f"{block_name}.{layer_idx}.weight"
                custom_bias_key = f"{block_name}.{layer_idx}.bias"
                if vgg_index < len(pretrained_keys):  # Ensure within range
                    pretrained_weight_key = pretrained_keys[vgg_index]
                    pretrained_bias_key = pretrained_keys[vgg_index + 1]
                    
                    # Check if the layers match in size before assigning
                    if dict.get(custom_weight_key) is None and dict.get(custom_bias_key) is None:
                        continue
                    if (dict[custom_weight_key].shape == weights[pretrained_weight_key].shape and
                        dict[custom_bias_key].shape == weights[pretrained_bias_key].shape):
                        print(f"Mapping: {custom_weight_key} -> {pretrained_weight_key}")
                        # Assign weights and biases
                        dict[custom_weight_key] = weights[pretrained_weight_key]
                        dict[custom_bias_key] = weights[pretrained_bias_key]
                    
                    # Move to the next layer in the VGG16 `features`
                    vgg_index += 2

        # Load the updated state dict into the custom model
        model.load_state_dict(dict)
    if n_classes != 1000:
        model.classifier[-1] = nn.Linear(4096, n_classes)
    return model


def vgg11_bn_f(pretrained=True,n_classes=100,ref_dict=None):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg["A"], batch_norm=True)
    if pretrained:
        print("Loading pretrained weights")
        # Load pretrained weights from torchvision
        weights = ref_dict
        # Load pretrained VGG16
        dict = model.state_dict()

        # Mapping logic: Directly map "features.*" layers to corresponding "block*.*" layers
        pretrained_keys = [k for k in weights.keys() if "features" in k]
        vgg_index = 0

        for block_name in ["block0", "block1", "block2", "block3", "block4"]:
            # Get the number of layers in each block from the custom model's configuration
            num_layers_in_block = len([k for k in dict.keys() if block_name in k])
            for layer_idx in range(num_layers_in_block):
                # Map VGG16 layer weights to corresponding custom model layer
                custom_weight_key = f"{block_name}.{layer_idx}.weight"
                custom_bias_key = f"{block_name}.{layer_idx}.bias"
                if vgg_index < len(pretrained_keys):  # Ensure within range
                    pretrained_weight_key = pretrained_keys[vgg_index]
                    pretrained_bias_key = pretrained_keys[vgg_index + 1]
                    
                    # Check if the layers match in size before assigning
                    if dict.get(custom_weight_key) is None and dict.get(custom_bias_key) is None:
                        continue
                    if (dict[custom_weight_key].shape == weights[pretrained_weight_key].shape and
                        dict[custom_bias_key].shape == weights[pretrained_bias_key].shape):
                        print(f"Mapping: {custom_weight_key} -> {pretrained_weight_key}")
                        # Assign weights and biases
                        dict[custom_weight_key] = weights[pretrained_weight_key]
                        dict[custom_bias_key] = weights[pretrained_bias_key]
                    
                    # Move to the next layer in the VGG16 `features`
                    vgg_index += 2

        # Load the updated state dict into the custom model
        model.load_state_dict(dict)
   
    return model

def vgg16_f(pretrained=True,n_classes=100,ref_dict=None):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["D"])
    if pretrained:
        print("Loading pretrained weights")
        # Load pretrained weights from torchvision
        weights = ref_dict
        # Load pretrained VGG16
        dict = model.state_dict()

        # Mapping logic: Directly map "features.*" layers to corresponding "block*.*" layers
        pretrained_keys = [k for k in weights.keys()]
        vgg_index = 0

        for block_name in ["block0", "block1", "block2", "block3", "block4"]:
            # Get the number of layers in each block from the custom model's configuration
            num_layers_in_block = len([k for k in dict.keys() if block_name in k])
            for layer_idx in range(num_layers_in_block):
                # Map VGG16 layer weights to corresponding custom model layer
                custom_weight_key = f"{block_name}.{layer_idx}.weight"
                custom_bias_key = f"{block_name}.{layer_idx}.bias"
                custom_running_mean_key = f"{block_name}.{layer_idx}.running_mean"
                custom_running_var_key = f"{block_name}.{layer_idx}.running_var"
                custom_num_batches_tracked_key = f"{block_name}.{layer_idx}.num_batches_tracked"
                print(
                    dict.get(custom_weight_key),dict.get(custom_bias_key),dict.get(custom_running_mean_key),dict.get(custom_running_var_key),dict.get(custom_num_batches_tracked_key)
                )
                count = 0 
                if vgg_index < len(pretrained_keys):  # Ensure within range
                    pretrained_weight_key = pretrained_keys[vgg_index]
                    pretrained_bias_key = pretrained_keys[vgg_index + 1]
                    
                    # Check if the layers match in size before assigning
                    if dict.get(custom_weight_key) is None and dict.get(custom_bias_key) is None:
                        continue
                    if (dict[custom_weight_key].shape == weights[pretrained_weight_key].shape and
                        dict[custom_bias_key].shape == weights[pretrained_bias_key].shape):
                        print(f"Mapping: {custom_weight_key} -> {pretrained_weight_key}")
                        # Assign weights and biases
                        dict[custom_weight_key] = weights[pretrained_weight_key]
                        dict[custom_bias_key] = weights[pretrained_bias_key]
                        count += 2
                    if dict.get(custom_running_mean_key) is not None and dict.get(custom_running_var_key) is not None:
                        dict[custom_running_mean_key] = weights[pretrained_keys[vgg_index + 2]]
                        dict[custom_running_var_key] = weights[pretrained_keys[vgg_index + 3]]
                        dict[custom_num_batches_tracked_key] = weights[pretrained_keys[vgg_index + 4]]
                        count += 3
                    
                    # Move to the next layer in the VGG16 `features`
                    vgg_index += count

        # Load the updated state dict into the custom model
        model.load_state_dict(dict)
    return model


def vgg16_bn_f(pretrained=True,n_classes=100,ref_dict=None):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg["D"],batch_norm=True)
    if pretrained:
        print("Loading pretrained weights")
        # Load pretrained weights from torchvision
        weights = ref_dict
        # Load pretrained VGG16
        dict = model.state_dict()

        # Mapping logic: Directly map "features.*" layers to corresponding "block*.*" layers
        pretrained_keys = [k for k in weights.keys()]
        vgg_index = 0

        for block_name in ["block0", "block1", "block2", "block3", "block4"]:
            # Get the number of layers in each block from the custom model's configuration
            num_layers_in_block = len([k for k in dict.keys() if block_name in k])
            for layer_idx in range(num_layers_in_block):
                # Map VGG16 layer weights to corresponding custom model layer
                custom_weight_key = f"{block_name}.{layer_idx}.weight"
                custom_bias_key = f"{block_name}.{layer_idx}.bias"
                custom_running_mean_key = f"{block_name}.{layer_idx}.running_mean"
                custom_running_var_key = f"{block_name}.{layer_idx}.running_var"
                custom_num_batches_tracked_key = f"{block_name}.{layer_idx}.num_batches_tracked"
                count = 0 
                if vgg_index < len(pretrained_keys):  # Ensure within range
                    pretrained_weight_key = pretrained_keys[vgg_index]
                    pretrained_bias_key = pretrained_keys[vgg_index + 1]
                    
                    # Check if the layers match in size before assigning
                    if dict.get(custom_weight_key) is None and dict.get(custom_bias_key) is None:
                        continue
                    if (dict[custom_weight_key].shape == weights[pretrained_weight_key].shape and
                        dict[custom_bias_key].shape == weights[pretrained_bias_key].shape):
                        print(f"Mapping: {custom_weight_key} -> {pretrained_weight_key}")
                        # Assign weights and biases
                        dict[custom_weight_key] = weights[pretrained_weight_key]
                        dict[custom_bias_key] = weights[pretrained_bias_key]
                        count += 2
                    if dict.get(custom_running_mean_key) is not None and dict.get(custom_running_var_key) is not None:
                        dict[custom_running_mean_key] = weights[pretrained_keys[vgg_index + 2]]
                        dict[custom_running_var_key] = weights[pretrained_keys[vgg_index + 3]]
                        dict[custom_num_batches_tracked_key] = weights[pretrained_keys[vgg_index + 4]]
                        count += 3
                    
                    # Move to the next layer in the VGG16 `features`
                    vgg_index += count

        classifier_layers = len([k for k in dict.keys() if 'classifier' in k])
        for cls_idx in [0,3,6]:
            custom_cls_key = f"classifier.{cls_idx}"
            pretrained_cls_key = f"classifier.{cls_idx}"
            dict[custom_cls_key + ".weight"] = weights[pretrained_cls_key + ".weight"]
            dict[custom_cls_key + ".bias"] = weights[pretrained_cls_key + ".bias"]
            print(f"Mapping Classifier layer: {custom_cls_key}")

        # Load the updated state dict into the custom model
        model.load_state_dict(dict)
    return model


def vgg19_f(pretrained=True,n_classes=100,ref_dict=None):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg["E"])
    if pretrained:
        print("Loading pretrained weights")
        # Load pretrained weights from torchvision
        weights = ref_dict
        # Load pretrained VGG16
        dict = model.state_dict()

        # Mapping logic: Directly map "features.*" layers to corresponding "block*.*" layers
        pretrained_keys = [k for k in weights.keys() if "features" in k]
        vgg_index = 0

        for block_name in ["block0", "block1", "block2", "block3", "block4"]:
            # Get the number of layers in each block from the custom model's configuration
            num_layers_in_block = len([k for k in dict.keys() if block_name in k])
            for layer_idx in range(num_layers_in_block):
                # Map VGG16 layer weights to corresponding custom model layer
                custom_weight_key = f"{block_name}.{layer_idx}.weight"
                custom_bias_key = f"{block_name}.{layer_idx}.bias"
                if vgg_index < len(pretrained_keys):  # Ensure within range
                    pretrained_weight_key = pretrained_keys[vgg_index]
                    pretrained_bias_key = pretrained_keys[vgg_index + 1]
                    
                    # Check if the layers match in size before assigning
                    if dict.get(custom_weight_key) is None and dict.get(custom_bias_key) is None:
                        continue
                    if (dict[custom_weight_key].shape == weights[pretrained_weight_key].shape and
                        dict[custom_bias_key].shape == weights[pretrained_bias_key].shape):
                        print(f"Mapping: {custom_weight_key} -> {pretrained_weight_key}")
                        # Assign weights and biases
                        dict[custom_weight_key] = weights[pretrained_weight_key]
                        dict[custom_bias_key] = weights[pretrained_bias_key]
                    
                    # Move to the next layer in the VGG16 `features`
                    vgg_index += 2

        # Load the updated state dict into the custom model
        model.load_state_dict(dict)
    if n_classes != 1000:
        model.classifier[-1] = nn.Linear(4096, n_classes)
    return model


def vgg19_bn_f(pretrained=True,n_classes=100,ref_dict=None):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg["E"], batch_norm=True)
    if pretrained:
        print("Loading pretrained weights")
        # Load pretrained weights from torchvision
        weights = ref_dict
        # Load pretrained VGG16
        dict = model.state_dict()

        # Mapping logic: Directly map "features.*" layers to corresponding "block*.*" layers
        pretrained_keys = [k for k in weights.keys()]
        vgg_index = 0

        for block_name in ["block0", "block1", "block2", "block3", "block4"]:
            # Get the number of layers in each block from the custom model's configuration
            num_layers_in_block = len([k for k in dict.keys() if block_name in k])
            for layer_idx in range(num_layers_in_block):
                # Map VGG16 layer weights to corresponding custom model layer
                custom_weight_key = f"{block_name}.{layer_idx}.weight"
                custom_bias_key = f"{block_name}.{layer_idx}.bias"
                custom_running_mean_key = f"{block_name}.{layer_idx}.running_mean"
                custom_running_var_key = f"{block_name}.{layer_idx}.running_var"
                custom_num_batches_tracked_key = f"{block_name}.{layer_idx}.num_batches_tracked"
                count = 0 
                if vgg_index < len(pretrained_keys):  # Ensure within range
                    pretrained_weight_key = pretrained_keys[vgg_index]
                    pretrained_bias_key = pretrained_keys[vgg_index + 1]
                    
                    # Check if the layers match in size before assigning
                    if dict.get(custom_weight_key) is None and dict.get(custom_bias_key) is None:
                        continue
                    if (dict[custom_weight_key].shape == weights[pretrained_weight_key].shape and
                        dict[custom_bias_key].shape == weights[pretrained_bias_key].shape):
                        print(f"Mapping: {custom_weight_key} -> {pretrained_weight_key}")
                        # Assign weights and biases
                        dict[custom_weight_key] = weights[pretrained_weight_key]
                        dict[custom_bias_key] = weights[pretrained_bias_key]
                        count += 2
                    if dict.get(custom_running_mean_key) is not None and dict.get(custom_running_var_key) is not None:
                        dict[custom_running_mean_key] = weights[pretrained_keys[vgg_index + 2]]
                        dict[custom_running_var_key] = weights[pretrained_keys[vgg_index + 3]]
                        dict[custom_num_batches_tracked_key] = weights[pretrained_keys[vgg_index + 4]]
                        count += 3
                    
                    # Move to the next layer in the VGG16 `features`
                    vgg_index += count

        classifier_layers = len([k for k in dict.keys() if 'classifier' in k])
        for cls_idx in [0,3,6]:
            custom_cls_key = f"classifier.{cls_idx}"
            pretrained_cls_key = f"classifier.{cls_idx}"
            dict[custom_cls_key + ".weight"] = weights[pretrained_cls_key + ".weight"]
            dict[custom_cls_key + ".bias"] = weights[pretrained_cls_key + ".bias"]
            print(f"Mapping Classifier layer: {custom_cls_key}")

        # Load the updated state dict into the custom model
        model.load_state_dict(dict)
    return model
    
    
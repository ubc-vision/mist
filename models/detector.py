import os
import torch
import torchvision.models
from itertools import product
from models.net import ResNet

class Detector(torch.nn.Module):
    def __init__(self, config):
        super(Detector, self).__init__()
        self.config = config
        if config.detector_backbone == 'ResNet34':
            print('init detector with ResNet')
            self.backbone = ResNet34Detector(config.pretrained_resnet)
        elif config.detector_backbone == 'CustomResNet':
            print('init detector with CustomResNet')
            self.backbone = CustomResNetDetector()
        else:
            raise NotImplementedError('{} backbone has not been implemented'.format(config.detector_backbone))
 
        scale = float(config.image_size)/self.backbone.get_stride()
        self.anchor_size = self.config.anchor_size*scale   
        

    def forward(self, images):
        diag = {}

        # Generate featuremap
        featuremap, heatmap = self.backbone(images,self.anchor_size)

        B,C,H,W = heatmap.shape

        # spatial softmax
        if self.config.spatial_softmax:
            sm_kernel_size = int(self.config.sm_kernal_size_ratio*(H+W)/4)*2+1
            heatmap[:,[0],:,:] = self.spatial_softmax(heatmap[:,[0],:,:], kernel_size=15, strength=10)

        # NMS
        nms_kernel_size = int(self.config.nms_kernal_size_ratio*(H+W)/4)*2+1
        heatmap_nms, _ = self.non_maximum_suppression(heatmap[:,[0],:,:], 5)
        # Choose top k
        _, indices = torch.topk(heatmap_nms.view(heatmap_nms.shape[0], -1), self.config.k)
        # Convert indices to x,y
        kp_int = torch.stack([(indices % W).float(),(indices // W).float()],dim=2)
        # Extract bbox from heatmap
        idx = torch.stack([torch.arange(images.shape[0]).reshape(-1,1,1).repeat(1,self.config.k,heatmap.shape[1]-1),
                           torch.arange(1,heatmap.shape[1]).reshape(1,1,-1).repeat(images.shape[0],self.config.k,1),
                           indices.unsqueeze(-1).repeat(1,1,heatmap.shape[1]-1)],dim=0) 
        bbox = heatmap.view(heatmap.shape[0],heatmap.shape[1],-1)[tuple(idx)]
        
        # add x,y to offset
        bbox[:,:,:2] = bbox[:,:,:2] + kp_int   

        # store results to dict
        diag['heatmap'] = heatmap
        diag['stride'] = self.backbone.get_stride()
        if self.config.patch_from_featuremap:
            diag['featuremap'] = featuremap.clone().detach().requires_grad_(False)
        else:
            diag['featuremap'] = images
        return bbox, diag

    def non_maximum_suppression(self, heatmap, size):
        # eps = 1e-4
        # heatmap = heatmap / eps
        max_logits = torch.nn.functional.max_pool2d(heatmap, kernel_size=size, stride=1, padding=size//2)
        mask = torch.ge(heatmap, max_logits)
        return heatmap * mask.float(), mask

    def spatial_softmax(self, heatmap, kernel_size, strength):

        # heatmap [N, S, H, W]
        out_shape = heatmap.shape
        heatmap = heatmap.view(-1, 1, out_shape[-2], out_shape[-1])
        pad = kernel_size // 2
        # max_logits = torch.nn.functional.max_pool2d(heatmap, kernel_size=kernel_size, stride=1)
        # max_logits = torch.nn.functional.pad(max_logits, pad=(pad, pad, pad, pad), mode='replicate')
        max_logits = torch.max(heatmap, 2, True)[0]
        max_logits = torch.max(max_logits, 3, True)[0]

        ex = torch.exp(strength * (heatmap - max_logits))
        # ex = torch.exp(strength * (heatmap))
        sum_ex = torch.nn.functional.avg_pool2d(ex, kernel_size=kernel_size, stride=1, count_include_pad=False) * kernel_size**2
        sum_ex = torch.nn.functional.pad(sum_ex, pad=(pad, pad, pad, pad), mode='replicate')
        probs = ex / (sum_ex + 1e-6)
        # probs = heatmap - max_logits
        probs = probs.view(*out_shape)
        return probs

    def get_featuremap_channels(self):
        if self.config.patch_from_featuremap:
            return self.backbone.get_featuremap_channels()
        else:
            return 3

# Resnet detector with 1by1 conv 
class ResNet34Detector(torch.nn.Module):
    def __init__(self, pretrained_resnet):
        super(ResNet34Detector, self).__init__()

        # not using the last maxpool layer
        self.backbone = torch.nn.Sequential(*list(self.create_resnet34(pretrained_resnet).children())[:7]) 

        for layer in range(len(self.backbone)):
            for p in self.backbone[layer].parameters(): p.requires_grad = False
        
        self.detector_head = torch.nn.Sequential(
            torch.nn.Conv2d(256, 32, 3, 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, 1, 1, 0),
        )

    def create_resnet34(self, pretrained_resnet):
        if pretrained_resnet:
            model = torchvision.models.resnet34(pretrained=True, progress=True)
        else:
            model = torchvision.models.resnet34(pretrained=False, progress=True)
        return model

    def get_featuremap_channels(self):
        return 256

    def get_stride(self):
        return 16

    def forward(self, images, anchor_size):

        featuremap = self.backbone(images)
        heatmap = self.detector_head(featuremap)

        B,_,W,H = heatmap.shape

        heatmap_w_h = torch.ones([B,2,W,H])*anchor_size
        heatmap_r = heatmap[:,[0],:,:]
        heatmap_x_y = torch.nn.functional.relu(heatmap[:,[1,2],:,:])
        heatmap = torch.cat([heatmap_r,heatmap_x_y,heatmap_w_h],dim=1)

        return featuremap, heatmap 


# Resnet detector with 1by1 conv 
class CustomResNetDetector(torch.nn.Module):
    def __init__(self, num_blocks=4, num_channels=32):
        super(CustomResNetDetector, self).__init__()
        self.pad = torch.nn.ReflectionPad2d(num_blocks * 2)
        self.norm = torch.nn.BatchNorm2d(3)
        self.resnet = ResNet(3, num_channels, num_blocks, use_padding=False)
        self.last = torch.nn.Conv2d(num_channels, 1, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, anchor_size):
        x = self.pad(x)
        x = self.norm(x)
        f_x = self.resnet(x)
        h_x = self.last(f_x)
        N,_,W,H = h_x.shape
        h_x = torch.cat([h_x,torch.zeros(N,2,W,H),torch.ones(N,2,W,H)*anchor_size],dim=1)
        return f_x, h_x

    def get_stride(self):
        return 1

    def get_featuremap_channels(self):
        return 32
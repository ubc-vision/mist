import torch
from models.spatialtransform import spatial_transformer
from models.net import ResNet


def ClassifierNetWork(config,feat_ch):
    if config.detector_backbone == 'CustomResNet':
        return CustomClassifier(config,feat_ch)
    elif config.detector_backbone == 'ResNet34':
        return ResnetClassifier(config,feat_ch)
    else:
        raise Exception('invalid backbone')

class ResnetClassifier (torch.nn.Module):
    
    def __init__(self, config,feat_ch):
        super(ResnetClassifier, self).__init__()


        self.psize = (config.patch_size, config.patch_size)
        self.cnn = ResNet(256, 512, 3, use_padding=True)
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(512,config.num_classes),
            torch.nn.Softmax(dim=-1))
        self.global_avg_pool = torch.nn.AvgPool2d((self.psize[0],self.psize[1]))
    
    def forward(self, featuremap, bboxes, stride=None):
        diag = {}

        B, N, _ = bboxes.shape
        _, C, _, _ = featuremap.shape

        patches = spatial_transformer(featuremap, bboxes, self.psize).view(-1, C, *self.psize)
        patches = self.cnn(patches)
        features = self.global_avg_pool(patches).view(patches.shape[0],-1)
        logits = self.fcn(features)
        logits = logits.view(B, N,-1)
        labels = torch.argmax(logits,dim=2)

        # get labels
        labels = torch.argmax(logits,dim=2)
        diag['patches'] = patches

        return labels, logits, diag



class CustomClassifier(torch.nn.Module):
    
    def __init__(self, config,feat_ch):
        super(CustomClassifier, self).__init__()
        self.encoder = Encoder(config)
        self.classifier = Classifier(config.num_classes, self.encoder.out_channels)
        self.psize = (config.patch_size, config.patch_size)
    
    def forward(self, featuremap, bboxes, stride=None):
        diag = {}

        B, N, _ = bboxes.shape
        _, C, _, _ = featuremap.shape

        # extract patch 
        patches = spatial_transformer(featuremap, bboxes, self.psize).view(-1, C, *self.psize)
        
        # compute logits
        latent = self.encoder(patches)
        logits = self.classifier(latent)
        logits = logits.view(B,N,-1)

        # get labels
        labels = torch.argmax(logits,dim=2)

        diag['patches'] = patches

        return labels, logits, diag

class Classifier(torch.nn.Module):
    def __init__(self, number_class,in_channels):
        super(Classifier, self).__init__()
        self.dense = torch.nn.Linear(in_channels, number_class)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.dense(x)
        x = self.softmax(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        num_channels = 8
        num_blocks = 3
        num_levels = 5

        c_out = 3
        self.layers = []
        for i in range(num_levels):
            c_in = c_out
            c_out = num_channels * 2**i
            self.layers += [ResNet(in_channels=c_in, num_channels=c_out, num_blocks=num_blocks)]
            self.layers += [torch.nn.MaxPool2d(2, stride=2)]
        self.out_channels = c_out
        self.layers += [ResNet(in_channels=c_out, num_channels=c_out, num_blocks=num_blocks)]
        self.layers = torch.nn.Sequential(*self.layers)
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], x.shape[1])
        return x

class ClassifierGAP (torch.nn.Module):
    
    def __init__(self, config,feat_ch):
        super(ClassifierGAP, self).__init__()

        self.psize = (config.patch_size, config.patch_size)
        self.cnn = ResNet(feat_ch, 512, 3, use_padding=True)
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(512,config.num_classes),
            torch.nn.Softmax(dim=-1))
        self.global_avg_pool = torch.nn.AvgPool2d((self.psize[0],self.psize[1]))
    
    def forward(self, featuremap, bboxes, stride=None):
        diag = {}

        B, N, _ = bboxes.shape
        _, C, _, _ = featuremap.shape

        # extract patch 
        patches = spatial_transformer(featuremap, bboxes, self.psize).view(-1, C, *self.psize)
        
        # compute logits
        patches = self.cnn(patches)
        features = self.global_avg_pool(patches).view(patches.shape[0],-1)
        logits = self.fcn(features)

        # get labels
        labels = torch.argmax(logits,dim=2)
        diag['patches'] = patches
        diag['features'] = features

        return labels, logits, diag







import torch

class ResNet(torch.nn.Module):
    def __init__(self, in_channels, num_channels, num_blocks, use_padding=True):
        super(ResNet, self).__init__()
        self.first = torch.nn.Conv2d(in_channels, num_channels, 1)
        self.blocks = []
        for _ in range(num_blocks):
            self.blocks += [ResNetBlock(num_channels, use_padding)]
        self.blocks = torch.nn.Sequential(*self.blocks)
    
    def forward(self, x):
        x = self.first(x)
        x = self.blocks(x)
        return x

class ResNetBlock(torch.nn.Module):
    def __init__(self, channels=32, use_padding=True):
        super(ResNetBlock, self).__init__()
        # self.norm = torch.nn.BatchNorm2d(channels)
        # self.norm = torch.nn.GroupNorm(1, channels) # group norm
        self.norm = torch.nn.GroupNorm(1, channels) # layer norm
        self.use_padding = use_padding
        
        pad = 1 if self.use_padding else 0
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=pad),
            self.norm,
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=pad),
            self.norm
        )
        
    def forward(self, x):
        x_in = x
        if not self.use_padding:
            x_in = x_in[:, :, 2:-2, 2:-2]
        return torch.nn.functional.relu(x_in + self.block(x))

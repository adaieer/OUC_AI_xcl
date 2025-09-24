import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # 重塑为 [batch, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    # 转置维度 [batch, channels_per_group, groups, height, width]
    x = torch.transpose(x, 1, 2).contiguous()
    
    # 展平回原始维度 [batch, channels, height, width]
    return x.view(batch_size, num_channels, height, width)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)
    
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class_num = 16

class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()
        # conv1：（1, 30, 25, 25）， 8个 7x3x3 的卷积核 ==>（8, 24, 23, 23）
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3))
        # conv2：（8, 24, 23, 23）， 16个 5x3x3 的卷积核 ==>（16, 20, 21, 21）
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3))
        # conv3：（16, 20, 21, 21），32个 3x3x3 的卷积核 ==>（32, 18, 19, 19）
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3))

        # 二维卷积：（576, 19, 19） 64个 3x3 的卷积核，得到 （64, 17, 17）
        self.conv4 = nn.Conv2d(576, 64, kernel_size=(3, 3))

        # 接下来依次为256，128节点的全连接层，都使用比例为0.4的 Dropout，
        self.fc1 = nn.Linear(18496, 256)
        self.fc2 = nn.Linear(256, 128)
        self.drop = nn.Dropout(p=0.4)
        # 最后输出为 16 个节点，是最终的分类类别数。
        self.out = nn.Linear(128, class_num)

    
    def forward(self, x):
        # x: (1, 1, 30, 25, 25)
        x = F.relu(self.conv1(x))  # (1, 8, 24, 23, 23)
        x = F.relu(self.conv2(x))  # (1, 16, 20, 21, 21)
        x = F.relu(self.conv3(x))  # (1, 32, 18, 19, 19)
        x = x.view(-1,x.shape[1]*x.shape[2],x.shape[3],x.shape[4])
        x = F.relu(self.conv4(x))
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)
        return x

# 随机输入，测试网络结构是否通
x = torch.randn(1, 1, 30, 25, 25)
net = HybridSN()
y = net(x)
print(y.shape)
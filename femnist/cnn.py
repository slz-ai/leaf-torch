import torch
import torch.nn as nn
from utils.model_utils import batch_data,unravel_model_params,ravel_model_params
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(in_features=7*7*64,out_features=2048),##in_feature计算一下
            nn.ReLU(),
            nn.Linear(in_features=2048,out_features=num_classes)
        )

    def forward(self,x):
        feature=self.conv(x)
        output=self.fc(feature.view(x.shape[0], -1))
        return output
    def get_params(self):##得到参数的值，格式要注意
        """
        Squash model parameters or gradients into a single tensor.
        """
        return ravel_model_params(self)
    def set_params(self,model_params):
        unravel_model_params(self,model_params)


def build_net(num_classes):
    net = CNN(num_classes)
    return net

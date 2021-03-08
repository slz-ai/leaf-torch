#import tensorflow as tf
import torch
from torch import nn
from model import Model
import numpy as np
from baseline_constants import ACCURACY_KEY, INPUT_SIZE
from utils.model_utils import build_net
class ClientModel(Model):
    def __init__(self,seed,dataset,model_name,count_ops,lr,num_classes):
        self.seed=seed
        self.model_name=model_name
        self.num_classes=num_classes
        self.dataset=dataset
        super(ClientModel,self).__init__(seed,lr,count_ops,count_ops=count_ops)
    def create_model(self):
        "建立模型返回 net loss trainer"
        net=build_net(self.dataset,self.model_name,self.num_classes)
        loss=nn.CrossEntropyLoss()
        trainer=torch.optim.SGD(net.parameters(),lr=self.lr)
        return net,loss,trainer
    def test(self, data):#可能得修改一下x_vec让他变成rorch
        x_vecs = self.preprocess_x(data['x'].detach().numpy())
        labels = self.preprocess_y(data['y'].detach().numpy())

        output = self.net(x_vecs)
        b=output.argmax(axis=1)
        index=0
        acc=0
        for item in b:
            if item==labels[index]:
                acc+=1
            index+=1
        acc=acc/index



        loss = self.losses(output, labels.type(torch.LongTensor)).detach().numpy().mean()
        return {ACCURACY_KEY: acc, "loss": loss}
    def preprocess_x(self, raw_x_batch):
        return torch.from_numpy(raw_x_batch.reshape((-1, *INPUT_SIZE)))

    def preprocess_y(self, raw_y_batch):
        return torch.from_numpy(raw_y_batch)








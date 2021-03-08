from abc import ABC, abstractmethod
import numpy as np
#import mxnet as mx
import warnings
import torch
#from mxnet import autograd
from utils.model_utils import batch_data,unravel_model_params,ravel_model_params
from baseline_constants import INPUT_SIZE

class Model(ABC):
    def __init__(self,seed,lr,ctx,optimizor=None,count_ops=None):
        self.seed=seed
        self.lr=lr
        self.ctx=ctx
        self._optimizor=optimizor
        self.count_ops=count_ops

        np.random.seed(123+self.seed)
        np.random.seed(self.seed)

        self.net, self.losses, self.trainer0 = self.create_model()

    @property
    def optimizer(self):
            """Optimizer to be used by the model."""
            if self._optimizer is None:
                self._optimizer = "sgd"

            return self._optimizer
    ###为啥什么都没有
    @abstractmethod
    def create_model(self):
        """Creates the model for the task.
        Returns:
            A 3-tuple consisting of:
                net: A neural network workflow.
                loss: An operation that, when run with the features and the
                    labels, computes the loss value.
                train_op: An operation that, when grads are computed, trains
                    the model.
        """
        return None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.
        Args:
            data: Dict of the form {'x': NDArray, 'y': NDArray}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for _ in range(num_epochs):
            print("---Epoch %d"%(_))
            self.run_epochs(data, batch_size)

        update = self.get_params()
        comp = num_epochs * len(data['y']) * self.flops \
            if self.count_ops else 0
        return comp, update

    def run_epochs(self, data, batch_size):
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            input_data = self.preprocess_x(batched_x)#这个要好好看看
            target_data = self.preprocess_y(batched_y)
            self.trainer0.zero_grad()
            y_hats = self.net(input_data)
            lss = self.losses(y_hats, target_data.type(torch.LongTensor))
            lss.backward()
            self.trainer0.step()

    def get_params(self):##得到参数的值，格式要注意
        """
        Squash model parameters or gradients into a single tensor.
        """
        return ravel_model_params(self.net)
    def set_params(self,model_params):
        unravel_model_params(self.net,model_params)



    @abstractmethod
    def test(self, data):
        """
        Tests the current model on the given data.
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        return None

    def __num_elems(self, shape):
        '''Returns the number of elements in the given shape
        Args:
            shape: Parameter shape

        Return:
            tot_elems: int
        '''
        tot_elems = 1
        for s in shape:
            tot_elems *= int(s)
        return tot_elems

    @abstractmethod
    def preprocess_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return None

    @abstractmethod
    def preprocess_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return None

    def size(self):
        '''Returns the size of the network in bytes
        The size of the network is calculated by summing up the sizes of each
        trainable variable. The sizes of variables are calculated by multiplying
        the number of bytes in their dtype with their number of elements, captured
        in their shape attribute
        Return:
            integer representing size of graph (in bytes)
        '''
        params = self.get_params().detach().numpy()
        tot_size = 0
        for p in params:
            tot_elems = self.__num_elems(p.shape)
            dtype_size = np.dtype(p.dtype).itemsize
            var_size = tot_elems * dtype_size
            tot_size += var_size
        return tot_size

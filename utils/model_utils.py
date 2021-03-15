import json
import numpy as np
import os
import importlib
import torch

from collections import defaultdict
from baseline_constants import INPUT_SIZE
def batch_data(data, batch_size, seed):
    """
    data is a dict := {'x': numpy array, 'y': numpy} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data['x'].detach().numpy()
    data_y = data['y'].detach().numpy()
    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir,test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    #print(train_clients)
    test_clients, test_groups, test_data = read_dir(test_data_dir)
    #print(test_clients)
    #print(test_data)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

def build_net(dataset,model_name,num_classes):
    model_file="%s.%s.py" %(dataset,model_name)
    if not os.path.exists(model_file):
        print("Please specify a valid model")
    model_path="%s.%s" %(dataset,model_name)
    #build net
    mod=importlib.import_module(model_path)
    build_net_op=getattr(mod,"build_net")#获得mod这个对象的build_net方法
    net=build_net_op(num_classes)


    return net
def ravel_model_params(model, grads=False, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    if cuda:
        m_parameter = torch.Tensor([0]).cuda()
    else:
        m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]
def unravel_model_params(model, parameter_update):
    """
    Assigns grad_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in grad_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from grad_update
    for p in model.parameters():
        numel = p.data.numel()
        size = p.data.size()
        p.data.copy_(parameter_update[current_index:current_index + numel].view(size))
        current_index += numel
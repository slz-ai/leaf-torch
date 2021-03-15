import os
import numpy as np
import torch
from utils.model_utils import build_net
from baseline_constants import BYTES_WRITTEN_KEY, \
    BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

class Server:
    def __init__(self, client_model, dataset, model_name, num_classes):
        self.dataset = dataset
        self.model_name = model_name
        self.selected_clients = []
        # build and synchronize the global model
        self.model = build_net(dataset, model_name, num_classes)
        self.model.set_params(client_model.get_params())

        self.updates=[]
        # build a model for merging updates
        self.merged_update = build_net(
            dataset, model_name, num_classes)
        self.total_weight = 0
    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:每个selected_client 的 num_train_samples num_test_samples
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(
            possible_clients, num_clients, replace=False)
        return [(c.num_train_samples, c.num_test_samples)
                for c in self.selected_clients]
    def train_model(self, num_epochs=1, batch_size=10, clients=None):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.
        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            bytes_written: number of bytes written by each client to server
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients

        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
            for c in clients}
        index=0
        for c in clients:###每个模型开始训练
            index+=1
            print("%d th clients start training"%index)
            c.model.set_params(self.model.get_params())
            comp, num_samples, update = c.train(num_epochs, batch_size) #update = self.get_params() (a tensor)
            #print("%d th clients start update"%index)
            #self.merge_updates(num_samples, update)
            #print("%d th clients finish merging" % index)
            self.updates.append((num_samples, update.detach().numpy()))
            #print("%d th clients finish update" % index)
            #print(self.updates)
            #print("%d th clients calculating metrices" % index)
            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size()
            #print("in size")
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size()
            #print("out size")
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
            #print("computation")

        return sys_metrics

    def merge_updates(self, client_samples, update):
        merged_update_ = list(self.merged_update.get_params())#tensor的列表
        #print(merged_update_)
        current_update_ = list(update)#list of tensor
        #print(current_update_)
        num_params = len(merged_update_)

        self.total_weight += client_samples#weight propotional with client_samples

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (client_samples * current_update_[p].data()))


    def update_model(self):
        print("start global updating")
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v)
        averaged_soln = [v / total_weight for v in base]#new parameters
        print("averaged_soln")
        #print(averaged_soln)
        self.model.set_params(torch.from_numpy(np.array(averaged_soln)))
        self.updates = []
        ##merged_update clear()

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model.get_params())
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics

        return metrics
    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples
    def save_model(self, log_dir):
        """Saves the server model on:
            logs/{self.dataset}/{self.model}.params
        """
        torch.save(self.model.state_dict(),os.path.join(log_dir, self.model_name + ".params"))

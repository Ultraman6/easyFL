"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
from flgo.algorithm.other.fedbase import BasicClient, BasicServer
from flgo.utils import fmodule
from flgo.utils.minimizers import SAGM


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rho': 0.1, 'alpha': 0.01})

class Client(BasicClient):
    @fmodule.with_multi_gpus
    def train(self, model):
        # global parameters
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = SAGM(model.parameters(), optimizer, model, self.alpha, self.rho)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.step()
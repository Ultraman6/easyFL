"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
import copy
from flgo.algorithm.fedbase import BasicServer, BasicClient
import torch
from flgo.utils import fmodule
from flgo.utils.fmodule import _model_to_tensor, deserialize_model, serialize_model
from flgo.utils.minimizers import SAM


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rho': 0.1})
        self.c = torch.zeros_like(_model_to_tensor(self.model))

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'delta': copy.deepcopy(self.c)
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        ws, cs = res['model'], res['delta_i']
        self.model = self.aggregate(ws)
        dc = _model_to_tensor(self.aggregate(cs) / self.num_clients)
        self.c += dc
        return len(ws) > 0

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.c_i = torch.zeros_like(_model_to_tensor(self.model))

    @fmodule.with_multi_gpus
    def train(self, model, c):
        # global parameters
        model.train()
        origin_param = _model_to_tensor(model)
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = SAM(optimizer, self.model, self.rho)

        for iter in range(self.num_steps):
            origin_model = copy.deepcopy(model)
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()

            self.calculator.compute_loss(model, batch_data)['loss'].backward()
            minimizer.ascent_step()
            self.calculator.compute_loss(model, batch_data)['loss'].backward()

            g_hat = serialize_model(model, 'grad.data')
            optimizer.zero_grad()
            grad = g_hat - self.c_i + c

            self.model.load(origin_model)
            deserialize_model(model, grad, flag='grad.data', mode='copy')
            optimizer.step()

        delta_c_i = (origin_param - _model_to_tensor(model)) / (self.learning_rate * self.num_steps) - c
        self.c_i += delta_c_i

        return delta_c_i

    def reply(self, svr_pkg):
        # model = self.unpack(svr_pkg)
        model, c = svr_pkg['model'], svr_pkg['delta']
        delta_c_i = self.train(model, c)
        cpkg = {
            "model" : model,
            "delta_i": delta_c_i,
        }
        return cpkg
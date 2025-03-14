"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
import copy
from flgo.algorithm.fedbase import BasicServer, BasicClient
import torch
from flgo.utils import fmodule
from flgo.utils.fmodule import _model_to_tensor, deserialize_model, serialize_model


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rho': 0.1})
        self.s = torch.zeros_like(_model_to_tensor(self.model))

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'delta': copy.deepcopy(self.s)
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        ws, ss = res['model'], res['delta_i']
        self.s = self.calc_s(ss)
        self.model = self.aggregate(ws)
        return len(ws) > 0

    def calc_s(self, parameters_list):
        weights = torch.ones(len(parameters_list)).cuda()
        weights = weights / torch.sum(weights)

        serialized_parameters = torch.sum(torch.stack(parameters_list, dim=-1) / weights, dim=-1)
        return self.rho * serialized_parameters / serialized_parameters.norm()

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.mu_i = torch.zeros_like(_model_to_tensor(self.model))

    @fmodule.with_multi_gpus
    def train(self, model, s):
        # global parameters
        model.train()
        hat_s = None
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        for iter in range(self.num_steps):
            origin_model = copy.deepcopy(model)
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            self.calculator.compute_loss(model, batch_data)['loss'].backward()
            tier = serialize_model(model) - self.mu_i - s
            hat_s = self.calc_hats(tier)
            self.mu_i += hat_s - s
            model.zero_grad()

            deserialize_model(model, tier, flag='data', mode='add')

            self.calculator.compute_loss(model, batch_data)['loss'].backward()
            self.model.load(origin_model)

            optimizer.step()

        return self.mu_i - hat_s

    def reply(self, svr_pkg):
        # model = self.unpack(svr_pkg)
        model, c = svr_pkg['model'], svr_pkg['delta']
        delta_c_i = self.train(model, c)
        cpkg = {
            "model" : model,
            "delta_i": delta_c_i,
        }
        return cpkg

    def calc_hats(self, tier):
        return self.rho * tier / tier.norm()

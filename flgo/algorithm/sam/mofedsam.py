"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
import copy
import numpy as np
from flgo.algorithm.fedbase import BasicClient, BasicServer
import torch
from flgo.utils import fmodule
from flgo.utils.fmodule import _model_to_tensor, deserialize_model
from flgo.utils.minimizers import MoSAM


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'eta_l': 0.01, 'eta_g': 1.0, 'rho':0.02, 'beta': 0.1})
        self.delta = torch.zeros_like(_model_to_tensor(self.model))

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'delta': copy.deepcopy(self.delta)
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        ws, ks = res['model'], res['step']
        self.delta = self.calc_momentum(ws, ks)
        serialized_parameters = _model_to_tensor(self.model) - self.delta * self.eta_g
        deserialize_model(self.model, serialized_parameters)

    def calc_momentum(self, ws, ks):
        K = np.array(ks).mean()
        gradient_list = [torch.sub(w, _model_to_tensor(self.model)) for w in ws]
        delta = torch.mean(torch.stack(gradient_list, dim=0), dim=0)
        delta.div_(-1*self.eta_l*K)
        return delta


class Client(BasicClient):

    @fmodule.with_multi_gpus
    def train(self, model, delta):
        # global parameters
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = MoSAM(optimizer, self.model, self.rho, self.beta, delta)
        num_steps = 0
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.ascent_step()

            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.descent_step()

            num_steps += 1
        return num_steps

    def reply(self, svr_pkg):
        # model = self.unpack(svr_pkg)
        model, delta = svr_pkg['model'], svr_pkg['delta']
        step = self.train(model, delta)
        return {"model" : model, 'step': step}
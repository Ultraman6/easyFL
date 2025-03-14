"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
from flgo.algorithm.fedbase import BasicClient, BasicServer
import torch
from flgo.utils import fmodule
from flgo.utils.fmodule import _model_to_tensor, _modeldict_norm
from flgo.utils.minimizers import LESAM_D

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rho': 0.1})
        self.s = torch.zeros_like(_model_to_tensor(self.model))

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.paramL = None
        self.register_cache_var('paramL')

    @fmodule.with_multi_gpus
    def train(self, model):
        # global parameters
        if self.paramL is None:
            perturb = - _model_to_tensor(model)
        else:
            self.paramL.to(self.device)
            perturb = self.paramL - _model_to_tensor(model)
        perturb /= _modeldict_norm(perturb, p=2)

        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = LESAM_D(optimizer, model, self.rho)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.ascent_step(perturb)

            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.descent_step()

        self.paramL = _model_to_tensor(model)
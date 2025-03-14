"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
import copy
from flgo.algorithm.fedbase import BasicServer, BasicClient
import torch
from flgo.utils import fmodule
from flgo.utils.fmodule import _model_to_tensor, deserialize_model
from flgo.utils.minimizers import SAM
from statistics import mean

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rho': 0.1, 'W': 10, 'T_D': 0.1, 'g_rho': 0.1})
        self.window = []
        self.c = 0
        self.pseudo_gradient = None
        self.perturbed_model_parameters = None

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'delta': copy.deepcopy(self.perturbed_model_parameters),
            'c': self.c
        }

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)
        pseudo_gradient = _model_to_tensor(self.model)
        self.model = self.aggregate(models)
        pseudo_gradient.sub_(_model_to_tensor(self.model))
        self.pseudo_gradient = pseudo_gradient
        self.calc_perturbation(pseudo_gradient)

        return len(models) > 0

    # check return value (int, float)
    def calc_c(self, buffer):
        Divergence_metric = torch.tensor([torch.norm(torch.sub(_model_to_tensor(self.model), ele)).item() for ele in buffer])
        tot_norm = torch.div(torch.sum(Divergence_metric), len(Divergence_metric)).item()
        self.append_grad_norm(tot_norm)
        self.norm_grad = tot_norm
        self.c = mean(self.window)

    def append_grad_norm(self, grad_norm):
        x = 1 if grad_norm > self.T_D else 0
        self.window.append(x)
        if len(self.window) > self.W:
            del (self.window[0])

    def calc_perturbation(self, grad):
        # Calculate the perturbation using parameters (always)
        self.perturbed_model_parameters = copy.deepcopy(_model_to_tensor(self.model))
        grad.div_(grad.norm(2)).mul_(self.g_rho)
        self.perturbed_model_parameters.add_(grad)

class Client(BasicClient):

    @fmodule.with_multi_gpus
    def train(self, model, delta, c):
        # global parameters
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = SAM(optimizer, self.model, self.rho)
        for iter in range(self.num_steps):
            init_model = None
            if delta is not None:
                init_model = _model_to_tensor(model)
            # get a batch of data
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.ascent_step()

            if delta is not None:
                self.weighted_sum(model, delta, c)

            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()

            if delta is not None:
                deserialize_model(model, init_model, flag='data', mode='copy')

            minimizer.descent_step()

    def weighted_sum(self, model, perturb_model, c):
        model_parameters = perturb_model*c + _model_to_tensor(model) * (1-c)
        deserialize_model(model, model_parameters, flag='data', mode='copy')
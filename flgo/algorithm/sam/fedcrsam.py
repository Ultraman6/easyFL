"""
This is a non-official implementation of 'Federated Optimization in Heterogeneous
Networks' (http://arxiv.org/abs/1812.06127)
"""
from flgo.algorithm.fedbase import BasicClient, BasicServer
from torch.nn.modules.batchnorm import _BatchNorm
from flgo.utils import fmodule
from flgo.utils.minimizers import CRSAM

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)
def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rho': 0.1, 'alpha': 0.1, 'beta': 0.01})

class Client(BasicClient):
    @fmodule.with_multi_gpus
    def train(self, model):
        # global parameters
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = CRSAM(optimizer, model, self.rho, self.alpha, self.beta)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            minimizer.first_step()

            batch_loss = self.calculator.compute_loss(model, batch_data)['loss']
            minimizer.o_l = batch_loss

            # compute gradient
            minimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            minimizer.first_step(zero_grad=True)

            # second forward-backward pass
            disable_running_stats(model)
            batch_loss = self.calculator.compute_loss(model, batch_data)['loss']

            optimizer.w_l = batch_loss

            batch_loss.backward(retain_graph=True)
            optimizer.second_step(zero_grad=True)

            # second forward-backward pass
            enable_running_stats(model)
            batch_loss = self.calculator.compute_loss(model, batch_data)['loss']

            optimizer.b_l = batch_loss

            batch_loss.backward()
            optimizer.third_step(zero_grad=True)

        return
import contextlib

import torch
from collections import defaultdict
from torch._C._distributed_c10d import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm


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



class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self, init_model=None):
        grad_record = []
        if init_model is None:
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                grad_record.append(p.grad.data.clone().flatten())
                p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        return torch.cat(grad_record)

class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                eps[...] = p.grad[...]
                self.state[p]["grad"] = eps.clone()
                eps.mul_(self.rho / grad_norm)
                self.state[p]["eps"] = eps
            p.add_(eps)
        self.optimizer.zero_grad()

    def get_perturbation(self):
        perturb = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            perturb.append(self.state[p]["eps"].clone().view(-1))
        return torch.cat(perturb)

    def get_gradient(self):
        perturb = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            perturb.append(self.state[p]["grad"].clone().view(-1))
        return torch.cat(perturb)


class MoSAM(SAM):
    def __init__(self, optimizer, model, rho, beta, delta):
        super().__init__(optimizer, model, rho)
        self.beta = beta
        self.delta = delta  # 全局-全局更新
        # self.model_parameters_np = model_parameters_np

    @torch.no_grad()
    def descent_step(self):
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape

            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])

            p.grad.mul_(self.beta)
            momentum_grad = self.delta[idx:idx + layer_size].view(shape)[:]
            momentum_grad = momentum_grad.mul_(1 - self.beta).cuda()

            p.grad.add_(momentum_grad)

            idx += layer_size
        self.optimizer.step()
        self.optimizer.zero_grad()

class NagSAM(SAM):
    def __init__(self, optimizer, model, rho, beta, delta_i):
        super().__init__(optimizer, model, rho)
        self.beta = beta
        self.delta_i = delta_i  # 全局-全局更新
        # self.model_parameters_np = model_parameters_np
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        # grads_record = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape
            if p.grad is None:
                continue
            # grads_record.append(p.grad.data.clone().view(-1))
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            if self.delta_i is not None:
                eps.mul_(self.beta)
                momentum_grad = self.delta_i[idx:idx + layer_size].view(shape)[:]
                momentum_grad = momentum_grad.mul_(1 - self.beta).cuda()
                eps.add_(momentum_grad)
            p.add_(eps)
            idx += layer_size
        self.optimizer.zero_grad()
        # return torch.cat(grads_record)

class GF_ADMM(SAM):
    @torch.no_grad()
    def descent_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


class LESAM(SAM):

    @torch.no_grad()
    def ascent_step(self, g_update):
        idx = 0
        for n, p in self.model.named_parameters():
            layer_size = p.grad.numel()
            shape = p.grad.shape
            eps = g_update[idx:idx + layer_size].view(shape)[:]
            eps = eps.mul_(self.rho).cuda()
            self.state[p]["eps"] = eps
            p.add_(eps)
            idx += layer_size
        self.optimizer.zero_grad()

class LESAM_D(SAM):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM_D, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def ascent_step(self, g_update):
        # first order sum
        grad_norm = 0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                else:
                    grad_norm += g_update[idx].norm(p=2)

        for group in self.param_groups:
            # if g_update !=None:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM

                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                else:
                    e_w = -g_update[idx] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w


class LESAM_S(SAM):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM_S, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def ascent_step(self, g_update):
        # first order sum
        grad_norm = 0
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                else:
                    grad_norm += g_update[idx].norm(p=2)

        for group in self.param_groups:
            # if g_update !=None:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None:
                    continue
                # original SAM
                # e_w = p.grad * scale.to(p)
                # ASAM

                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                else:
                    e_w = -g_update[idx] * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w

class CRSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.1, alpha=0.1, beta=0.1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(CRSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.beta = beta
        self.o_l = 0
        self.w_l = 0
        self.b_l = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self.grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["original"] = p.grad.clone()
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["worst"] = p.grad.clone()
                p.sub_(self.state[p]["e_w"] * 2.0)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["best"] = p.grad.clone()
                p.add_(self.state[p]["e_w"])

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.grad = self.state[p]["worst"] + self.alpha * (self.state[p]["worst"] + self.state[p]["best"] - 2 * self.state[p]["original"]) + self.beta * ((self.state[p]["worst"] - self.state[p]["best"]))

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, alpha, rho, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_t = rho
        self.perturb_eps = perturb_eps
        self.alpha = alpha

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        # self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["f"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['f'] * p.grad.data
                )
                self.state[p]['p'] = p.grad.data.clone()
                self.state[p]['h'] = p.grad.data - self.state[p]['f']
                self.state[p]['hf'] = self.state[p]['h'] - self.state[p]['f']

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='f')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['f'] - cosine * old_grad_norm * p.grad.data / (
                            new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def get_grads(self):
        p, f, h, hf = [], [], [], []
        for key in self.state.keys():
            p.append(self.state[key]['p'].clone().flatten())
            f.append(self.state[key]['f'].clone().flatten())
            h.append(self.state[key]['h'].clone().flatten())
            hf.append(self.state[key]['hf'].clone().flatten())
        return torch.cat(p), torch.cat(f), torch.cat(h), torch.cat(hf)

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value


class SAGM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, alpha, rho, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAGM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_t = rho
        self.perturb_eps = perturb_eps
        self.alpha = alpha

        # initialize self.rho_t
        self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        # self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = (rho / (grad_norm + self.perturb_eps) - self.alpha)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sam_grad = self.state[p]['old_g'] * 0.5 - p.grad * 0.5
                p.grad.data.add_(sam_grad)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:

            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        else:

            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )

        return norm

    # def norm(tensor_list: List[torch.tensor], p=2):
    #     """Compute p-norm for tensor list"""
    #     return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value
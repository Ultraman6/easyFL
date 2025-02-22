import copy

import numpy as np
from itertools import chain
import torch
from torch.utils.data import DataLoader, Dataset
from wandb.cli.cli import server
from flgo.experiment.logger import BasicLogger
from flgo.utils.metrics.frobenius import frobenius_norm
from flgo.utils.metrics.hessian import grad_norm, hessian_trace
from flgo.utils.metrics.pac_bayes import pac_bayes_bound
from flgo.utils.metrics.path_norm import path_norm
from flgo.utils.minimizers import SAM


class VirtualDataLoader(DataLoader):
    def __init__(self, loaders, **kwargs):
        """
        使用多个 DataLoader 的数据，创建一个虚拟的 DataLoader。

        Args:
            loaders (list[DataLoader]): 多个 DataLoader。
            kwargs: DataLoader 的其他参数（如 batch_size、shuffle 等）。
        """
        self.loaders = loaders
        self.iterator = chain(*[iter(loader) for loader in loaders])  # 合并迭代器
        self._length = sum(len(loader) for loader in loaders)  # 合并后的总批次数

        # 初始化时传入一个空的 Dataset，避免强制要求 Dataset 类型
        super().__init__(dataset=self._virtual_dataset(), **kwargs)

    def __iter__(self):
        """
        返回合并后的迭代器。
        """
        return self.iterator

    def __len__(self):
        """
        返回合并后的总批次数。
        """
        return self._length

    @staticmethod
    def _virtual_dataset():
        """
        创建一个虚拟的 Dataset，只用于满足 DataLoader 初始化需求。
        """
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0
            def __getitem__(self, idx):
                raise IndexError("This dataset is virtual and should not be accessed directly.")
        return EmptyDataset()

class LandLogger(BasicLogger):
    r"""Simple Logger. Only evaluating model performance on testing dataset and validation dataset."""
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        self.metrics = ['frobenius_norm', 'pac_bayes_bound', 'path_norm', 'normalized_trace', 'approximate_ratio']

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        self.info('Current_time:{}'.format(self.clock.current_time))
        self.output['time'].append(self.clock.current_time)
        self._test()
        self._val()

        global_model = copy.deepcopy(server.model).train()
        global_criterion = server.calculator.criterion
        global_optimizer = server.calculator.get_optimizer(global_model, lr=server.learning_rate,
                                                    weight_decay=server.options['weight_decay'],
                                                    momentum=server.options['momentum'])
        device = server.device

        for metric in self.metrics:
            self.output['val_' + metric] = self._land_metric(global_model, self.val_dl, global_criterion, global_optimizer, device, metric)
        for metric in self.metrics:
            self.output['test_' + metric] = self._land_metric(global_model, self.test_dl, global_criterion, global_optimizer, device, metric)
        self._flatness_discrepancy(global_model, global_criterion, global_optimizer, device)
        self._perturbation_drifts(global_model)

    @property
    def val_dl(self):
        local_dl = []  # 为了不修改 baseclient 类的妥协写法
        for c in self.participants:
            if c._train_loader is None:
                c._train_loader = c.calculator.get_dataloader(c.train_data, batch_size=c.batch_size,
                                                                    num_workers=c.loader_num_workers,
                                                                    pin_memory=c.option['pin_memory'],
                                                                    drop_last=c.option.get('drop_last', False))
            local_dl.append(c._train_loader)
        return VirtualDataLoader(local_dl)

    @property
    def test_dl(self):
        if not hasattr(self, '_test_dl'):
            test_batch_size = min(self.option['test_batch_size'], len(server.test_data))
            self._test_dl = server.calculator.get_dataloader(server.test_data, batch_size=test_batch_size,
                                                                        num_workers=server.option['num_workers'],
                                                                        pin_memory=server.option['pin_memory'],
                                                                        drop_last=server.option.get('drop_last', False))
        return self._test_dl

    def _test(self):
        test_metric = server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)

    def _val(self):
        val_metrics = server.global_test(flag='val')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in val_metrics.items():
            self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met
             for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)

    def _perturbation_drifts(self, global_model):
        # 扰动范数漂流(felesam)
        local_gradients = []
        local_perturbs = []
        for c in self.participants:
            local_model = copy.deepcopy(c.model).train()
            local_criterion = c.calculator.criterion
            local_optimizer = c.calculator.get_optimizer(local_model, lr=c.learning_rate,
                                                        weight_decay=c.options['weight_decay'],
                                                        momentum=c.options['momentum'])
            optimizer = c.calculator.get_optimizer(global_model, lr=c.learning_rate,
                                                        weight_decay=c.options['weight_decay'],
                                                        momentum=c.options['momentum'])
            local_device = c.device
            local_perturbs.append(self._land_metric(local_model, c._train_loader, local_criterion,
                                                     local_optimizer, local_device, 'perturbation'))
            local_gradients.append(self._land_metric(global_model, c._train_loader, local_criterion,
                                                     optimizer, local_device, 'gradient'))
        global_gradient = torch.sum(torch.stack(local_gradients), dim=0)
        global_perturb = global_gradient * (server.options['rho'] / torch.norm(global_gradient, p=2))
        self.output['perturbation_drifts'] = np.mean([torch.norm(global_perturb - lp, p=2) for lp in local_perturbs]) / 2

    def _flatness_discrepancy(self, global_model, global_criterion, global_optimizer, device):
        # 平坦性差异(fedgf)
        global_flatness = self._land_metric(global_model, self.test_dl, global_criterion, global_optimizer, device, 'flatness')
        w = 1 / len(self.participants)
        for c in self.participants:
            local_model = copy.deepcopy(c.model).train()
            local_criterion = c.calculator.criterion
            local_optimizer = c.calculator.get_optimizer(local_model, lr=c.learning_rate,
                                                        weight_decay=c.options['weight_decay'],
                                                        momentum=c.options['momentum'])
            local_device = c.device
            local_flatness = self._land_metric(local_model, c._train_loader, local_criterion, local_optimizer, local_device, 'flatness')
            global_flatness -= w * local_flatness
        self.output['flatness_discrepancy'] = global_flatness

    def _land_metric(self, model, data_loader, criterion, optimizer, device, mode: str):
        if mode == 'frobenius_norm':
            return frobenius_norm(
                model=model,
                data_loader=data_loader,
                criterion=criterion,
                device=device
            )
        elif mode == 'pac_bayes_bound':
            return pac_bayes_bound(
                model=model,
                data_loader=data_loader,
                criterion=criterion,
                device=device
            )
        elif mode == 'path_norm':
            return path_norm(
                model=model,
                data_loader=data_loader
            )
        elif mode == 'normalized_trace':
            trace = hessian_trace(
                model=model,
                dataloader=data_loader,
                criterion=criterion
            )
            norm = grad_norm(
                model=model,
                dataloader=data_loader,
                criterion=criterion,
                optimizer=optimizer
            )
            return trace / norm
        else:
            minimizer = SAM(optimizer, model, self.coordinator.options['rho'])
            values = []
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                model.zero_grad()
                loss_origin = criterion(model(inputs), labels)
                loss_origin.backward()
                minimizer.ascent_step()
                if mode == 'perturbation':
                    values.append(minimizer.get_perturbation())
                    continue
                elif mode == 'gradient':
                    values.append(minimizer.get_gradient())
                    continue
                model.zero_grad()
                loss_one_step = criterion(model(inputs), labels)
                if mode == 'flatness':
                    values.append(loss_one_step.item() - loss_origin.item())
                    continue
                elif mode == 'approximate_ratio':
                    loss_one_step.backward()
                    minimizer.ascent_step()
                    for _ in range(19): # 梯度上升-累计扰动
                        model.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        minimizer.ascent_step()
                    loss_optim = criterion(model(inputs), labels)
                    values.append((loss_one_step.item() - loss_origin.item()) / (loss_optim.item() - loss_origin.item()))
            if mode in ['perturbation', 'gradient']: # perturb tensor gradient tensor
                return torch.mean(torch.stack(values), dim=0)
            return np.mean(values)





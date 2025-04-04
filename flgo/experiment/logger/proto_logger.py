from flgo.experiment.logger import BasicLogger
import os, torch
import pickle

class ProtoLogger(BasicLogger):
    r"""Simple Logger. Only evaluating model performance on testing dataset and validation dataset."""
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))
        self.file_root = str(os.path.join(self.task_path, self.get_time_string() + self.get_output_name('')))
        os.makedirs(self.file_root, exist_ok=True)
        self.save_data_criterion(-1)
        for c in self.participants:
            self.save_data_criterion(c.id)

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        self.info('Current_time:{}'.format(self.clock.current_time))
        self.output['time'].append(self.clock.current_time)
        self.output['round'].append(self.current_round)
        # 1. 全局测试集
        test_metric = self.coordinator.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # 2. 本地训练/验证集(联合)
        val_metrics = self.coordinator.global_test(flag='train')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in val_metrics.items():
            self.output['val_'+met_name+'_dist'].append(met_val)
            self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # 3. 保存算子(全局/本地)
        self.save_model(-1)
        for c in self.participants:
            self.save_model(c.id)
        # 记录至文件
        self.show_current_output()
        self.save_output_as_json()

    def save_data_criterion(self, idx: int=-1):
        obj = self.participants[idx] if idx >= 0 else self.coordinator
        file_path = os.path.join(self.file_root, f'client{idx}' if idx >= 0 else 'global')
        os.makedirs(file_path, exist_ok=True)

        train_data_path = os.path.join(file_path, 'train_data.pkl')
        train_X, train_Y = [], []
        for x, y in obj.train_data:
            train_X.append(x)
            train_Y.append(y)
        pickle.dump((train_X, train_Y), open(train_data_path, 'wb'))

        test_data_path = os.path.join(file_path, 'test_data.pkl')
        if idx == -1:
            test_X, test_Y = [], []
            for x, y in obj.test_data:
                test_X.append(x)
                test_Y.append(y)
            pickle.dump((test_X, test_Y), open(test_data_path, 'wb'))

        criterion_path = os.path.join(file_path, 'criterion.pkl')
        pickle.dump(obj.calculator.criterion, open(criterion_path, 'wb'))

    def save_model(self, idx: int=-1):
        obj = self.participants[idx] if idx >= 0 else self.coordinator
        file_path = os.path.join(self.file_root, f'client{idx}' if idx >= 0 else 'global')
        os.makedirs(file_path, exist_ok=True)
        pickle.dump(obj.model, open(os.path.join(file_path, f'round={self.current_round}.pkl'), 'wb'))
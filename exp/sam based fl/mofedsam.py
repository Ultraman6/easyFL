import os
import wandb
import flgo.algorithm.sam.mofedsam as mofedsam
import flgo.experiment.analyzer as fea
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
import flgo.experiment.analyzer
from flgo.experiment.analyzer import merge_records
from flgo.experiment.logger.simple_logger import SimpleLogger

# 'rho': [0.02, 0.05, 0.1] proportion': [0.05, 0.1, 0.2]
min_fix = ['loss']
max_fix = ['accuracy']
metrics = ['round',
           'val_loss', 'val_accuracy', 'val_frobenius_norm', 'val_pac_bayes_bound', 'val_path_norm', 'val_normalized_trace', 'val_approximate_ratio',
           'test_loss', 'test_accuracy', 'test_frobenius_norm', 'test_pac_bayes_bound', 'test_path_norm', 'test_normalized_trace', 'test_approximate_ratio',
           'flatness_discrepancy', 'perturbation_drifts']
random_seeds = [0, 1, 42, 100, 666, 1024, 1334, 2025, 3407, 114514]
project = 'sam based fl based fl'
partition = 'cifar10_dir0.5_sample100_ratio0.1'
task = os.path.join('../tasks', project, partition, 'seed')
os.makedirs(task, exist_ok=True)

def run(seed: int=0):
    _task = os.path.join(task, str(seed))
    flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(
        num_clients=100,
        alpha=0.5
    ), task_path=_task, overwrite=False, seed=seed)

    runner = flgo.init(_task, mofedsam,
    {'seed': seed, 'dataseed':seed, 'gpu':0, 'log_file':True,
            'num_epochs':1, 'num_rounds': 1, 'eval_interval': 1,
            'batch_size': 64, 'learning_rate': 0.01, 'weight_decay': 0.0004, 'momentum': 0.0,
            'proportion': 0.1, 'parallel_type': 't', 'num_parallel': 20, 'sample_record_mode': 'w',
            'eta_l': 0.1, 'eta_g': 0.1, 'rho': 0.2, 'beta': 0.1,
    }, Logger=SimpleLogger)

    runner.run()

    return fea.load_records(_task, 'mofedsam')

if __name__ == '__main__':
    records = {}
    final_records = []
    # 1. 场景-种子-算法
    for seed in random_seeds:
        rs = run(seed)
        for r in rs: # 方法(算法+超参)存在
            if r.data['label'] not in records:
                records[r.data['label']] = []
            records[r.data['label']].append(r)
    # 2. 收集
    for label, rs in records.items():
        record = merge_records(rs, task=task, flag='seed', metrics=metrics)
        record._save()
        final_records.append(record)
        # wandb
        wandb.init(project=project, name=record.name, config=dict(record.option))
        for i, r in enumerate(record.data[metrics[0]]):
            wandb.log({m: record.data[m][i] for m in metrics[1:]}, step=r)
    # 3. 展示
    table = fea.Table(final_records)
    for m in metrics[1:]:
        v = fea.mean_value
        if m in min_fix:
            v = fea.min_value
        elif m in max_fix:
            v = fea.max_value
        table.add_column(v, {'x':m})
    table.print()


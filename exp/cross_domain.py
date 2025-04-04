import os
from itertools import product
import concurrent.futures
import wandb
import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer as fea
import flgo.benchmark.domainnet_classification as domain_net
# import flgo.benchmark.pacs_classification as pacs
# import flgo.benchmark.officehome_classification as office_home
import flgo.benchmark.partition as fbp
from flgo.benchmark.toolkits.visualization import visualize_by_domain_class
import flgo.experiment.analyzer
from flgo.experiment.logger.simple_logger import SimpleLogger

project = 'domainNet'
metrics = ['round', 'val_loss', 'val_accuracy', 'test_loss', 'test_accuracy']
alpha_list = [0, 0.2, 0.5, 100]

def run(label_alpha: int, domain_alpha: int):
    partition = f'dir{label_alpha}_dir{domain_alpha}'
    task = os.path.join('../tasks', project, partition)
    os.makedirs(task, exist_ok=True)

    domain_net.visualize = visualize_by_domain_class
    flgo.gen_task_by_(domain_net,
    fbp.LabelDomainPartitioner(
        partitioner_label=fbp.DirichletPartitioner(num_clients=100, alpha=label_alpha),
        partitioner_domain=fbp.DirichletPartitioner(num_clients=100, alpha=domain_alpha),
    ), task_path=task)

    # running fedavg on the specified task
    runner = flgo.init(task, fedavg, {
        'num_epochs': 5, 'num_rounds': 10000, 'eval_interval': 800, 'batch_size': 64,
        'learning_rate': 0.01, 'weight_decay': 0.0004, 'momentum': 0.0,
        'proportion': 0.1, 'parallel_type': 't', 'num_parallel': 20,
    }, Logger=SimpleLogger)
    runner.run()

    # visualize the experimental result
    records = fea.load_records(task, ['fedavg'], {'num_epochs':1})
    table = fea.Table(records)
    table.add_column(fea.min_value, {'x':'val_loss'})
    table.add_column(fea.min_value, {'x':'val_accuracy'})
    table.add_column(fea.min_value, {'x':'test_loss'})
    table.add_column(fea.min_value, {'x':'test_accuracy'})
    table.print()

    analysis_plan = {
        'Selector': {  # 选择可视化参与算法
            'task': task,
            'header': ['fedavg']
        },
        'Painter': {  # 过程记录-一站式可视化
            'Curve': [
                {'args': {'x': 'communication_round', 'y': 'val_loss'},
                 'fig_option': {'title': 'valid loss on Synthetic'}},
                {'args': {'x': 'communication_round', 'y': 'val_accuracy'},
                 'fig_option': {'title': 'valid accuracy on Synthetic'}},
                {'args': {'x': 'communication_round', 'y': 'test_loss'},
                 'fig_option': {'title': 'test accuracy on cifar10'}},
                {'args': {'x': 'communication_round', 'y': 'test_accuracy'},
                 'fig_option': {'title': 'test accuracy on cifar10'}},
            ]
        }
    }
    flgo.experiment.analyzer.show(analysis_plan)
    wandb.init(project=project, name=partition, config=dict(records[0].option))
    for i, r in enumerate(records[0].data[metrics[0]]):
        wandb.log({m: records[0].data[m][i] for m in metrics[1:]}, step=r)


if __name__ == '__main__':
    # 创建线程池（max_workers 可根据需要调整）
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务到线程池，可以使用 itertools.product 生成笛卡尔积
        futures = [executor.submit(run, a, b) for a, b in product(alpha_list, repeat=2)]
        # 如果需要等待所有线程执行完毕，并处理返回结果（如果 run 函数有返回值）
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                # 如有需要，可以在这里对 result 进行处理
            except Exception as e:
                print(f"任务出现异常: {e}")
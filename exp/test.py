import os, sys
root = os.path.dirname(os.getcwd())
sys.path.append(root)
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp

root = '../../records/easyfl'
task = os.path.join(root, 'test_parallel')
os.makedirs(root, exist_ok=True)
if __name__ == '__main__':
    # generate federated task (remark: if task already exists, this line will not work)
    flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=20, alpha=100), task_path=task)
    runner = flgo.init(task, fedavg, {'gpu':0, 'log_file':True, 'num_epochs':1, 'eval_interval': 10,
                                      'proportion': 1, 'parallel_type': 'p', 'num_parallels': 20,
                                      'train_holdout': 0.0})
    runner.run()
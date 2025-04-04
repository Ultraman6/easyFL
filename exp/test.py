import os, sys
root = os.path.dirname(os.getcwd())
sys.path.append(root)
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp

task = './my_task'
if __name__ == '__main__':
    # generate federated task (remark: if task already exists, this line will not work)
    flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=100), task_path=task)
    runner = flgo.init(task, fedavg, {'gpu':0, 'log_file':True, 'num_epochs':1})
    runner.run()
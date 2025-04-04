import os, sys, argparse, importlib
root = os.path.dirname(os.getcwd())
sys.path.append(root)
import flgo
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
from flgo.experiment.logger.simple_logger import SimpleLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=str(os.path.join(root, 'results', 'proto_based_fl')))
    parser.add_argument('--alpha', nargs='+', type=float, default=[0.05, 0.5, 100])
    parser.add_argument('--method', type=str, default='fedproto')
    parser.add_argument('--gpu', nargs='+', default=0)
    parser.add_argument('--num_rounds', nargs='+',default=1000)
    parser.add_argument('--eval_interval', nargs='+',default=10)
    parser.add_argument('--num_epochs', nargs='+', default=1)
    parser.add_argument('--num_clients', nargs='+',default=20)
    parser.add_argument('--proportion', nargs='+',default=1.0)
    parser.add_argument('--train_holdout', nargs='+',default=0)
    parser.add_argument('--learning_rate', nargs='+',default=0.01)
    parser.add_argument('--learning_rate_decay', nargs='+',default=0.998)
    parser.add_argument('--lr_scheduler', nargs='+',default='-1')
    parser.add_argument('--optimizer', nargs='+',default='SGD')
    parser.add_argument('--batch_size', nargs='+',default=10)
    parser.add_argument('--weight_decay', nargs='+',default=0.0004)
    parser.add_argument('--parallel_type', nargs='+',default='t')
    parser.add_argument('--num_parallel', nargs='+',default=10)
    args = parser.parse_args()

    for alpha in args.alpha:
        os.makedirs(args.task, exist_ok=True)
        task_path = str(os.path.join(args.task, f'simple_log_dir{alpha}'))
        print(f"开始运行 alpha = {alpha} 的实验，任务路径为: {task_path}")
        flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=args.num_clients, alpha=alpha), task_path=task_path)
        algo_class = importlib.import_module(f'flgo.algorithm.{args.method}')
        runner = flgo.init(task_path, algo_class, vars(args), Logger=SimpleLogger)
        print(f"成功初始化 runner for alpha = {alpha}")
        history = runner.run()
        print(f"成功运行 alpha = {alpha} 的实验")
import argparse
import torch
import torch.optim as optim
import ray

from ray import train
from ray.air import RunConfig
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

from components.model import DigitClassifier
from components.train_loop import train_model, evaluate_model


def train_job(config):
    """
    Runs distributed training job with Ray
    """
    train_loader = train.torch.prepare_data_loader(
        data_loader=config['train_loader'],
        add_dist_sampler=True,
        move_to_device=True,
        auto_transfer=True
    )

    test_loader = train.torch.prepare_data_loader(
        data_loader=config['test_loader'],
        add_dist_sampler=True,
        move_to_device=True,
        auto_transfer=True
    )

    model = DigitClassifier()
    model = train.torch.prepare_model(
        model=model,
        move_to_device=train.torch.get_device(),
        parallel_strategy=config['parallel_strategy']
    )
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    optimizer = train.torch.prepare_optimizer(optimizer)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for epoch in range(1, config['epochs']+1):
        train_model(model, train_loader, optimizer, epoch)
        evaluate_model(model, test_loader)
    end.record()
    print(f'Training Time Elapsed: {start.elapsed_time(end) / 1000}')


def get_args():
    """
    Get addtional command-line arguments for training job
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--parallel_strategy', type=str, default='ddp')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import json
    from components.dataset import get_data_loaders

    with open('config.json') as f:
        config = json.load(f)   

    scaling_config = ScalingConfig(
        num_workers=config['scaling_cfg']['num_workers'],
        use_gpu=config['scaling_cfg']['use_gpu'],
        trainer_resources={'CPU': config['scaling_cfg']['num_cpu'],}
    )

    run_config = RunConfig(name=config['run_cfg']['experiment_name'])

    torch_config = train.torch.TorchConfig(
        backend=config['torch_cfg']['backend'],
        timeout_s=config['torch_cfg']['timeout'],
    )

    train_loader, test_loader = get_data_loaders(batch_size=config['train_cfg']['batch_size'])

    trainer = TorchTrainer(
        train_loop_per_worker=train_job,
        train_loop_config={
            'train_loader': train_loader,
            'batch_size': config['train_cfg']['batch_size'],
            'epochs': config['train_cfg']['epochs'],
            'learning_rate': config['train_cfg']['learning_rate'],
            'parallel_strategy': config['train_cfg']['parallel_strategy']
        },
        torch_config=torch_config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    results = trainer.fit()
    print(results.metrics)
    print(results.checkpoint)
import torch


def parse_args():
    """arguments"""
    config = {
        "seed": 0,
        "device": torch.device("mps"),
        "batch_size": 32,
        "num_epoch": 5,
        "n_workers": 2,
        "learning_rate": 1e-4,
        "logging_step": 100,
        "warmup_steps": 1000
    }

    return config
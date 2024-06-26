import sys
from enum import Enum, auto

import clearml
import ignite
import numpy as np
import torch
from clearml import Logger
from ignite.contrib.handlers.clearml_logger import *
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint
from ignite.metrics import Accuracy, Loss, Fbeta, Precision
from ignite.metrics import ConfusionMatrix
from ignite.metrics.recall import Recall
from snntorch import spikegen
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import v2

from models import IzhikevichNet
from utils import SpikeDensityHandler


class Dataset(Enum):
    MNIST = auto()
    FASHION_MNIST = auto()


def load_data(transform, batch_size, dataset: Dataset):
    if (dataset == Dataset.MNIST):
        return load_mnist(transform, batch_size)
    elif (dataset == Dataset.FASHION_MNIST):
        return load_fashion_mnist(transform, batch_size)
    else:
        raise ValueError("No valid dataset was provided")


def load_mnist(transform, batch_size):
    data_path = '/tmp/data/mnist'
    try:
        data = clearml.datasets.Dataset.get(dataset_name="MNIST", dataset_version="1.0.0")
        data_path = data.get_local_copy()
        data_train = MNIST(data_path, train=True, download=False, transform=transform)
        data_test = MNIST(data_path, train=False, download=False, transform=transform)
    except Exception:
        data_train = MNIST(data_path, train=True, download=True, transform=transform)
        data_test = MNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def load_fashion_mnist(transform, batch_size):
    data_path = '/tmp/data/fashion_mnist'
    try:
        data = clearml.datasets.Dataset.get(dataset_name="FashionMNIST", dataset_version="1.0.0")
        data_path = data.get_local_copy()
        data_train = FashionMNIST(data_path, train=True, download=False, transform=transform)
        data_test = FashionMNIST(data_path, train=False, download=False, transform=transform)
    except Exception:
        data_train = FashionMNIST(data_path, train=True, download=True, transform=transform)
        data_test = FashionMNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def loss(prediction, targets, **kwargs):
    loss_val = torch.zeros(1, device=device)
    mem_rec = prediction[1]
    num_steps = len(mem_rec)
    for step in range(num_steps):
        loss_val += nn.CrossEntropyLoss()(mem_rec[step], targets)

    return loss_val[0] / num_steps


def output_transform(output):
    y_pred, y = output
    _, idx = y_pred[0].sum(dim=0).max(1)
    y_pred = ignite.utils.to_onehot(idx.long(), 10)
    return y_pred, y


def attatch_logging_handlers(trainer):
    val_metrics = {
        "accuracy": Accuracy(output_transform=output_transform),
        "cm": ConfusionMatrix(output_transform=output_transform, num_classes=10),
        "loss": Loss(criterion),
        "F1": Fbeta(beta=1, output_transform=output_transform),
        "recall": Recall(output_transform=output_transform, average=True),
        "precision": Precision(output_transform=output_transform, average=True),
    }

    train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        cm = train_evaluator.state.metrics['cm']
        Logger.current_logger().report_confusion_matrix(
            "train confusion",
            "train",
            iteration=trainer.state.iteration,
            matrix=np.matrix(cm),
            xaxis="Predicted Labels",
            yaxis="True Labels",
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        cm = val_evaluator.state.metrics['cm']
        Logger.current_logger().report_confusion_matrix(
            "test confusion",
            "test",
            iteration=trainer.state.iteration,
            matrix=np.matrix(cm),
            xaxis="Predicted Labels",
            yaxis="True Labels",
        )

    clearml_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=50),
        tag="training",
        output_transform=lambda loss: {"batch loss": loss},
    )

    clearml_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation metrics",
        metric_names=["loss", "accuracy", "recall", "F1", "precision"],
        global_step_transform=global_step_from_engine(trainer),
    )

    clearml_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="spike activation",
        metric_names=["spike_activity_layer_1", "spike_activity_layer_2"],
        global_step_transform=global_step_from_engine(trainer),
    )

    clearml_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training metrics",
        metric_names=["loss", "accuracy", "recall", "F1", "precision"],
        global_step_transform=global_step_from_engine(trainer),
    )

    checkpoint_handler = Checkpoint(
        {"model": model},
        ClearMLSaver(),
        n_saved=3,
        score_function=lambda e: e.state.metrics["accuracy"],
        score_name="val_acc",
        filename_prefix="best",
        global_step_transform=global_step_from_engine(trainer),
    )

    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
    clearml_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED
    )

    clearml_logger.attach(
        trainer,
        log_handler=WeightsScalarHandler(model),
        event_name=Events.EPOCH_COMPLETED
    )

    clearml_logger.attach(
        trainer,
        log_handler=GradsScalarHandler(model),
        event_name=Events.EPOCH_COMPLETED
    )

    clearml_logger.attach(
        trainer,
        log_handler=SpikeDensityHandler(model),
        event_name=Events.ITERATION_COMPLETED(every=25)
    )


class ToSpikes():
    def __init__(self, num_steps, gain=0.2):
        self.num_steps = num_steps
        self.gain = gain

    def __call__(self, x):
        return spikegen.rate(x, num_steps=self.num_steps, gain=self.gain, )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps}, gain={self.gain})"


class DirectCoding():
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, x):
        return torch.stack([x] * self.num_steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps})"


config = {
    "lr": 0.01,
    "num_steps": 128,
    "batch_size": 32,
    "neuron_type": "RS",
    "max_epochs": 50,
    "alpha": 0.95,
    "beta": 0.85,
    "dataset": Dataset.FASHION_MNIST,
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = ClearMLLogger(task_name="IZH base task without psp fashion", project_name="Masterarbeit/Izhikevich")
    clearml_logger.get_task().connect(config)
    model = IzhikevichNet(num_steps=config["num_steps"],
                          num_input=28 * 28,
                          neuron_type=config["neuron_type"],
                          alpha=config["alpha"],
                          beta=config["beta"],
                          use_psp=False).to(device)

    if config["alpha"] <= config["beta"]:
        clearml_logger.get_task().mark_completed(True, "invalid config", force=True)
        sys.exit("invalid config")

    transform = v2.Compose([
        v2.Resize((28, 28)),
        v2.Grayscale(),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(-1)),
        DirectCoding(num_steps=config["num_steps"]),
    ])

    train_loader, val_loader = load_data(transform, config["batch_size"], dataset=config["dataset"])


    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, 1)
            m.bias.data.fill_(0.1)


    model.apply(init_weights)
    print(model.eval())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    criterion = loss

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    attatch_logging_handlers(trainer)

    trainer.run(train_loader, max_epochs=config["max_epochs"])
    clearml_logger.get_task().completed()

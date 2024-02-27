import clearml
import ignite
import numpy as np
import torch
from clearml import Logger, OutputModel
from ignite.contrib.handlers.clearml_logger import *
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint
from ignite.metrics import Accuracy, Loss, Fbeta, Precision
from ignite.metrics import ConfusionMatrix
from ignite.metrics.recall import Recall
from snntorch import spikegen
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import v2
import sys

sys.path.append("../")

from models.fc_snn import SimpleFC
from models.conv_lif2 import SimpleCNN
import models.vgg as vgg
import models.bntt2 as bntt


def binary_one_hot_output_transform(output):
    y_pred, y = output
    y_pred = ignite.utils.to_onehot(y_pred.long(), 10)
    y = y.long()
    return y_pred, y


def load_data(transform, batch_size):
    data_path = '/home/hans/src/Masterarbeit/src/notebooks/cifar10'
    #try:
    #data = clearml.datasets.Dataset.get(dataset_name="cifar100", dataset_version="1.0.0")
    #data_path = data.get_local_copy()
    #data_train = CIFAR10(data_path, train=True, download=False, transform=transform)
    #data_test = CIFAR10(data_path, train=False, download=False, transform=transform)
    #except Exception:
    data_train = CIFAR10(data_path, train=True, download=True, transform=transform)
    data_test = CIFAR10(data_path, train=False, download=True, transform=transform)
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


class ToSpikes():
    def __init__(self, num_steps, gain=0.2):
        self.num_steps = num_steps
        self.gain = gain

    def __call__(self, x):
        return spikegen.rate(x, num_steps=self.num_steps, gain=self.gain, )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps}, gain={self.gain})"


config = {
    "num_steps": 25,
    "beta": 0.9,
    "gain": 0.1,
    "batch_size": 64,
    "max_epochs": 30,
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = ClearMLLogger(task_name="bntt CIFAR10", project_name="Masterarbeit")
    clearml_logger.get_task().connect(config)
    #model = SimpleCNN(config["beta"], num_steps=config["num_steps"]).to(device)
    #model = vgg.vgg11(config["num_steps"], batch_norm=True).to(device)
    img_size = 32
    num_cls = 10
    #model = bntt.SNN_VGG11_BNTT(num_steps=config["num_steps"], leak_mem=config["beta"], img_size=img_size, num_cls=num_cls).to(device)
    model = bntt.bntt(config["num_steps"], num_cls=num_cls).to(device)

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((32, 32)),
        v2.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ToSpikes(num_steps=config["num_steps"], gain=config["gain"]),
    ])

    train_loader, val_loader = load_data(transform, config["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = loss

    trainer = create_supervised_trainer(model, optimizer, criterion, device )
    attatch_logging_handlers(trainer)

    trainer.run(train_loader, max_epochs=config["max_epochs"])
    clearml_logger.get_task().completed()

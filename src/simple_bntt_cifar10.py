import ignite
import numpy as np
import torch
from clearml import Logger
from ignite.contrib.handlers.clearml_logger import *
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint
from ignite.metrics import ConfusionMatrix
from ignite.metrics import Loss, Fbeta, Precision, TopKCategoricalAccuracy
from ignite.metrics.recall import Recall
from snntorch import spikegen
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

import models.bntt as bntt


def load_data(transform, test_transform, batch_size):
    data_path = '/notebooks/cifar10'
    data_train = CIFAR10(data_path, train=True, download=True, transform=transform)
    data_test = CIFAR10(data_path, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader


def loss(mem_rec, targets, **kwargs):
    return nn.CrossEntropyLoss()(mem_rec, targets)


def attatch_logging_handlers(trainer):
    val_metrics = {
        "accuracy": TopKCategoricalAccuracy(output_transform=output_transform, k=1),
        "cm": ConfusionMatrix(output_transform=output_transform, num_classes=10),
        "loss": Loss(criterion),
        "F1": Fbeta(output_transform=output_transform, beta=1),
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


class DirectCoding():
    def __init__(self, num_steps, gain=0.2):
        self.num_steps = num_steps
        self.gain = gain

    def __call__(self, x):
        return torch.stack([x] * self.num_steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps})"


def adjust_learning_rate(optimizer, cur_epoch, max_epoch):
    if cur_epoch < 4:
        for param_group in optimizer.param_groups:
            param_group['lr'] += 0.04
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 1.1


config = {
    "num_steps": 25,
    "beta": 0.99,
    "batch_size": 80,
    "max_epochs": 100,
    "lr": 0.1,
    "architecture": "VGG11",
    "BNTT": True,
}


def output_transform(output):
    return output
    y_pred, y = output
    _, idx = y_pred[0].sum(dim=0).max(1)
    y_pred = ignite.utils.to_onehot(idx.long(), 10)
    return y_pred, y


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = ClearMLLogger(task_name="myBNTT + Dropout - test augmentation",
                                   project_name="Masterarbeit/Test")
    clearml_logger.get_task().connect(config)
    img_size = 32
    num_cls = 10
    model = bntt.vgg_with_bntt(cfg=config["architecture"],
                               num_steps=config["num_steps"],
                               batch_norm=config["BNTT"],
                               beta=config["beta"],
                               num_cls=num_cls
                               ).to(device)

    print(model.eval())
    transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.822, 0.4465), (0.2023, 0.1994, 0.2010)),
        DirectCoding(num_steps=config["num_steps"])
    ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.822, 0.4465), (0.2023, 0.1994, 0.2010)),
        DirectCoding(num_steps=config["num_steps"])
    ])

    train_loader, val_loader = load_data(transform, test_transform, config["batch_size"])

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = loss

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    attatch_logging_handlers(trainer)


    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_lr(trainer: ignite.engine.Engine):
        adjust_learning_rate(optimizer, trainer.state.epoch, trainer.state.max_epochs)


    trainer.run(train_loader, max_epochs=config["max_epochs"])
    clearml_logger.get_task().completed()

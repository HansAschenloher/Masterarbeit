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
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import sys

sys.path.append("../")

from models.fc_snn import SimpleFC


def binary_one_hot_output_transform(output):
    y_pred, y = output
    y_pred = ignite.utils.to_onehot(y_pred.long(), 10)
    y = y.long()
    return y_pred, y


def load_data(transform, batch_size):
    data_path = '/tmp/data/mnist'
    try:
        data = clearml.datasets.Dataset.get(dataset_name="mnist", dataset_version="1.0.0")
        data_path = data.get_local_copy()
        mnist_train = MNIST(data_path, train=True, download=False, transform=transform)
        mnist_test = MNIST(data_path, train=False, download=False, transform=transform)
    except Exception:
        mnist_train = MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = MNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

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

    @trainer.on(Events.ITERATION_COMPLETED(every=50))
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

    @trainer.on(Events.ITERATION_COMPLETED(every=50))
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

    for tag, evaluator in [("training metrics", train_evaluator), ("validation metrics", val_evaluator)]:
        clearml_logger.attach_output_handler(
            evaluator,
            event_name=Events.ITERATION_COMPLETED(every=50),
            tag=tag,
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
    "num_steps": 64,
    "beta": 0.9,
    "gain": 0.1,
    "batch_size": 128,
    "max_epochs": 3,
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = ClearMLLogger(task_name="simple snn", project_name="Masterarbeit")
    clearml_logger.get_task().connect(config)
    model = SimpleFC(config["beta"], num_steps=config["num_steps"]).to(device)

    transform = v2.Compose([
        v2.Resize((28, 28)),
        v2.Grayscale(),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0,), (1,)),
        v2.Lambda(lambda x: x.view(-1)),
        ToSpikes(num_steps=config["num_steps"], gain=config["gain"]),
    ])

    train_loader, val_loader = load_data(transform, config["batch_size"])

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
    criterion = loss

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    attatch_logging_handlers(trainer)


    trainer.run(train_loader, max_epochs=config)
    clearml_logger.get_task().completed()

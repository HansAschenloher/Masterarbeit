import numpy as np
from clearml import Logger
from ignite.contrib.handlers.clearml_logger import ClearMLSaver, OptimizerParamsHandler, WeightsScalarHandler, \
    GradsScalarHandler
from ignite.engine import Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Accuracy, ConfusionMatrix, Fbeta, Loss, Recall, Precision

from utils import SpikeDensityHandler


def attach_logging_handlers(clearml_logger, trainer, criterion, train_loader, val_loader, model, output_transform,
                            optimizer, device):
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
        event_name=Events.ITERATION_COMPLETED(every=50)
    )

import torch
from ignite.contrib.handlers.clearml_logger import *
from ignite.engine import create_supervised_trainer
from torch import nn
from torchvision.transforms import v2

from neurons import NeuronModels
from utils import Dataset, load_data, attach_logging_handlers


class SimpleFCNet(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()

        layers = []
        num_in = layer_config[0]
        for i, layer in enumerate(layer_config[1:]):
            if isinstance(layer, int):
                layers.append(nn.Linear(num_in, layer))
                layers.append(nn.ReLU(inplace=True))
                num_in = layer

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def loss(prediction, targets, **kwargs):
    return nn.CrossEntropyLoss()(prediction, targets)


def output_transform(output):
    # y_pred, y = output
    # y = ignite.utils.to_onehot(y, 10)
    return output


config = {
    "lr": 0.01,
    "batch_size": 128,
    "max_epochs": 10,
    "dataset": Dataset.FASHION_MNIST,
    "neuron_model": NeuronModels.RELU,
    "layer_config": [28 * 28, 100, 10]
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = ClearMLLogger(task_name="RELU base fashion_mnist", project_name="Masterarbeit/RELU")
    clearml_logger.get_task().connect(config)

    model = SimpleFCNet(
        layer_config=config["layer_config"],
    ).to(device)

    transform = v2.Compose([
        v2.Resize((28, 28)),
        v2.Grayscale(),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(-1)),
    ])

    train_loader, val_loader = load_data(transform, config["batch_size"], dataset=config["dataset"])


    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, 1)
            m.bias.data.fill_(0.1)


    model.apply(init_weights)
    print(model.eval())

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    trainer = create_supervised_trainer(model, optimizer, loss, device)
    attach_logging_handlers(clearml_logger=clearml_logger,
                            trainer=trainer,
                            criterion=loss,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            model=model,
                            output_transform=output_transform,
                            optimizer=optimizer,
                            device=device
                            )

    trainer.run(train_loader, max_epochs=config["max_epochs"])
    clearml_logger.get_task().completed()

import ignite
import snntorch as snn
import torch
from ignite.contrib.handlers.clearml_logger import *
from ignite.engine import create_supervised_trainer
from torch import nn
from torchvision.transforms import v2

import sys
sys.path.append("..")

from neurons import NeuronModels
from spike_generation import DirectCoding
from utils import Dataset, load_data, attach_logging_handlers


class SimpleFCNet(nn.Module):
    def __init__(self, alpha, beta, num_steps, layer_config, reset_mechanism='zero', **kwargs):
        super().__init__()
        self.num_steps = num_steps

        layers = []
        num_in = layer_config[0]
        for i, layer in enumerate(layer_config[1:]):
            if isinstance(layer, int):
                layers.append(nn.Linear(num_in, layer))
                if i == len(layer_config) - 2:
                    layers.append(
                        snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True, output=True, reset_mechanism="none", log_spikes=False)
                    )
                else:
                    layers.append(snn.Synaptic(alpha=alpha, beta=beta, init_hidden=True, output=False, reset_mechanism=reset_mechanism,
                                            log_spikes=True))
                num_in = layer

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net:
            if isinstance(layer, snn.SpikingNeuron):
                layer.reset_mem()

        spk_rec = []
        mem_rec = []
        for step in range(self.num_steps):
            spk, syn, mem = self.net(x[:, step])
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)


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


config = {
    "lr": 0.01,
    "num_steps": 16,
    "batch_size": 128,
    "max_epochs": 10,
    "alpha": 0.9,
    "beta": 0.8,
    "dataset": Dataset.FASHION_MNIST,
    "neuron_model": NeuronModels.SYNAPTIC,
    "reset_mechanism": "zero",
    "layer_config": [28 * 28, 100, 10]
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clearml_logger = ClearMLLogger(task_name="Synaptic base fashion_mnist", project_name="Masterarbeit/Synaptic")
    clearml_logger.get_task().connect(config)

    model = SimpleFCNet(num_steps=config["num_steps"],
                        layer_config=config["layer_config"],
                        reset_mechanism=config["reset_mechanism"],
                        alpha=config["alpha"],
                        beta=config["beta"],
                        ).to(device)

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

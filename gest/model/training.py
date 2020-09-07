import argparse

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from gest.model import IMAGE_WIDTH, IMAGE_HEIGHT


class BinaryClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.example_input_array = torch.randn(size=(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
        self.net = torch.nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_HEIGHT // 2 ** 3) * (IMAGE_WIDTH // 2 ** 3), 64), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1), nn.Sigmoid(),
            nn.Flatten(0),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.binary_cross_entropy(self(x), y.float())
        return pl.TrainResult(loss)

    def configure_optimizers(self):
        return Adam(self.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Data path")
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = ImageFolder(
        args.data_path,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(.05, 0), shear=5),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ]),
    )
    model = BinaryClassifier()
    trainer = pl.Trainer(logger=False, max_epochs=args.epochs)
    trainer.fit(model, DataLoader(dataset, 32, shuffle=True))
    torch.onnx.export(
        model,
        model.example_input_array,
        args.model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

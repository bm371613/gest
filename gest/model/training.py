import argparse

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from gest.model import CLASS_COUNT, IMAGE_WIDTH, IMAGE_HEIGHT
from gest.model.dataset import Dataset


def vectorize_target(target):
    class_index, additional_annotations = target
    return torch.Tensor([
        class_index,
        0 if additional_annotations is None else additional_annotations
    ])


class Model(pl.LightningModule):

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
            nn.Linear(64, 4), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    def loss(self, batch):
        x, y = batch
        class_indexes = y[:, 0].long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat[:, :CLASS_COUNT], class_indexes)
        mask = class_indexes == 2
        loss += F.mse_loss(y_hat[mask, CLASS_COUNT:], y[mask, 1:])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def configure_optimizers(self):
        return Adam(self.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Data path")
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = Dataset(
        args.data_path,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(.05, 0), shear=5),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
        ]),
        target_transform=vectorize_target,
    )
    model = Model()
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

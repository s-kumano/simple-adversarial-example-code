from typing import Literal, Union

import torch
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchvision.transforms.functional import normalize
from tqdm import tqdm


@torch.no_grad()
def in_range(x: Tensor, min: float, max: float) -> bool:
    return ((min<=x.min()) & (x.max()<=max)).item() # type: ignore


def read_labels(path: str) -> list[str]:
    with open(path, 'r') as f:
        labels = [label.replace('\n', '') for label in f.readlines()]
    return labels


def freeze(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool, 
    num_workers: int = 3, 
    pin_memory: bool = True, 
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def set_seed(seed: int = 0) -> None:
    seed_everything(seed, True)


class ModelWithNormalization(Module):
    def __init__(self, model: Module, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        assert in_range(x, 0, 1)
        return self.model(normalize(x, self.mean, self.std))
    

class CalcClassificationAcc(LightningLite):
    def run(
        self, 
        classifier: Module, 
        loader: DataLoader, 
        n_class: int, 
        top_k: int = 1,
        average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro',
    ) -> Union[float, list[float]]:

        classifier = self.setup(classifier)
        loader = self.setup_dataloaders(loader) # type: ignore

        freeze(classifier)
        classifier.eval()

        metric = MulticlassAccuracy(n_class, top_k, average)
        self.to_device(metric)

        for xs, labels in tqdm(loader):
            outs = classifier(xs)
            metric(outs, labels)
        
        acc = metric.compute()
        return acc.tolist() if average == 'none' else acc.item()
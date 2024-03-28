from typing import Literal, Optional, Union

import cv2
import torch
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchvision.models import ResNeXt50_32X4D_Weights, resnext50_32x4d
from torchvision.transforms.functional import normalize
from tqdm import tqdm


@torch.no_grad()
def in_range(x: Tensor, min: float, max: float) -> bool:
    return ((min<=x.min()) & (x.max()<=max)).item() # type: ignore


def read_labels(path: str) -> list[str]:
    with open(path, 'r') as f:
        labels = [label.replace('\n', '') for label in f.readlines()]
    return labels


def setup_model(device: Optional[Union[str, torch.device]] = None) -> Module:
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model = ModelWithNormalization(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model = model.eval()
    freeze(model)
    if device is not None:
        model.to(device)
    return model


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


def save_torch_img(path: str, img: Tensor) -> None:
    assert in_range(img, 0, 1)
    npimg = img.cpu().permute(1, 2, 0).numpy() * 255
    npimg = npimg.astype('uint8')
    npimg = cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR) # type: ignore
    cv2.imwrite(path, npimg) # type: ignore


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
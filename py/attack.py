import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as TF
import torchvision
import torchvision.transforms.functional as VF
from autoattack.autopgd_base import APGDAttack

root = os.path.join(os.path.sep, 'root', 'project')
sys.path.append(root)

from utils.apgd import APGDTargeted
from utils.utils import read_labels, save_torch_img, set_seed, setup_model


def main(
    img_name: str, 
    perturbation_size: int, 
    target_class: Optional[int], 
    img_size: int,
    device_id: Optional[int],
) -> None:
    
    device = f'cuda:{device_id}' if isinstance(device_id, int) else 'cpu'
    set_seed()

    # make dir
    img_name_stem = os.path.splitext(img_name)[0]
    out_root = os.path.join(root, 'advs', img_name_stem)
    os.makedirs(out_root, exist_ok=True)

    # setup model
    model = setup_model(device)

    # ImageNet labels
    label_path = os.path.join(root, 'labels', 'imagenet_labels.en.txt')
    imagenet_labels = read_labels(label_path)

    # read clean img
    img_path = os.path.join(root, 'imgs', img_name)
    clean_img = torchvision.io.read_image(img_path)
    clean_img = VF.center_crop(clean_img, min(clean_img.shape[1:])) # type: ignore
    clean_img = VF.resize(clean_img, [img_size, img_size], antialias=True)
    clean_img = clean_img / 255

    # prediction for clean img
    clean_img = clean_img.unsqueeze(0).to(device)
    clean_prob = TF.softmax(model(clean_img), dim=1)
    clean_idx_tensor = clean_prob.max(dim=1).indices
    clean_idx = clean_idx_tensor.item()
    clean_label = imagenet_labels[clean_idx] # type: ignore
    clean_prob = clean_prob[0][clean_idx] # type: ignore
    print(clean_idx, clean_label, f'{clean_prob:.2f}')

    # save clean img
    clean_fname = f'perturbation=0_resolution={img_size}' \
                  + f'_{clean_idx}_{clean_label}_{clean_prob:.2f}.png'
    clean_path = os.path.join(out_root, clean_fname)
    save_torch_img(clean_path, clean_img[0])

    # atk
    eps = perturbation_size / 255
    if isinstance(target_class, int):
        atk = APGDTargeted(model, eps)
        label = torch.tensor([target_class], device=device)
        adv = atk.perturb(clean_img, label).detach()
    else:
        atk = APGDAttack(model, eps=eps)
        label = clean_idx_tensor
        adv = atk.perturb(clean_img, label, best_loss=True).detach()

    # prediction for adv img
    adv_prob = TF.softmax(model(adv), dim=1)
    adv_idx_tensor = adv_prob.max(dim=1).indices
    adv_idx = adv_idx_tensor.item()
    adv_label = imagenet_labels[adv_idx] # type: ignore
    adv_prob = adv_prob[0][adv_idx] * 100 # type: ignore
    print(adv_idx, adv_label, f'{adv_prob:.2f}')

    # save adv
    adv_fname = f'perturbation={perturbation_size}_resolution={img_size}' \
                + f'_target={target_class}_{adv_idx}_{adv_label}_{adv_prob:.2f}.png'
    adv_path = os.path.join(out_root, adv_fname)
    save_torch_img(adv_path, adv[0])

    # check
    print((adv-clean_img).norm(float('inf')).item()*255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_name')
    parser.add_argument('perturbation_size', type=int)
    parser.add_argument('--target_class', '-t', type=int)
    parser.add_argument('--img_size', '-i', default=224, type=int)
    parser.add_argument('--device_id', '-d', type=int)
    args = parser.parse_args()

    main(args.img_name, args.perturbation_size, args.target_class, args.img_size, args.device_id)
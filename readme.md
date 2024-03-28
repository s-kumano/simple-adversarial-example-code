This repository includes code related to adversarial examples, attacks, and defenses. This repository does not contain novel contents. It serves as a reference for beginners interested in this field.

# Setup
Please run either of the following.

## Docker
If you can use GPUs,

```console
docker-compose -f "docker/docker-compose.gpu.yaml" up -d --build 
```

If you can use only CPUs,

```console
docker-compose -f "docker/docker-compose.cpu.yaml" up -d --build 
```

Note that it is difficult to run `ipynbs/movie/*.ipynb` in a docker container due to the authentication of a camera device.

## pyenv
This setup assumes the running of `ipynbs/movie/*.ipynb`. You may also be able to run other `ipynb` and `py` files with this setup. However, this has not been verified.

```console
pyenv install 3.11.8
pyenv global 3.11.8
pip install \
  matplotlib==3.8.3 \
  seaborn==0.13.2 \
  opencv-python==4.8.1.78 \
  torch==2.1.2 \
  torchvision==0.16.2 \
  pytorch-lightning==1.9.5 \
  lightning-lite==1.8.6 \
  ipykernel==6.29.3 \
  ipython==8.22.2 \
  torchattacks==3.5.1
```

## NOTE
I will destructively update this repository, so files created in the past may not be executable with the current setup. Please refer to `Dockerfile.old` for the past setup.

# Contents
- `imgs`: Clean images.
- `advs`: Adversarial examples created by `bash/attack.sh` and `py/attack.py`.
- `ipynbs/movie/*`: Prediction for objects in a movie. You can test physical adversarial examples such as adversarial patches.
- `ipynbs/attack.ipynb`: AutoPGD.
- `ipynbs/check.ipynb`: Check the perturbation norm of adversarial images in `advs`.
- `ipynbs/comparison.ipynb`: Comparison with FGSM, PGD, and APGD in CIFAR-10 and comparison with cross-entropy and DLR loss under gradient masking.
- `ipynbs/width.ipynb`: Comparison with adversarially trained MLP models across various network widths.
- `py/attack.py`: AutoPGD.
- `utils/apgd.py`: AutoPGD targeting a specific class. This is not commonly implemented in adversarial attack libraries such as [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) and [autoattack](https://github.com/fra31/auto-attack).
- `utils/mlp.py`: A simple MLP model with or without shortcuts.
- `weights`: Adversarially trained MLP weights across various network widths.
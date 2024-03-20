This repository includes code related to adversarial examples, attacks, and defenses. This repository does not contain novel contents. It serves as a reference for beginners interested in this field.

# Setup
Please run either of the following.

## Docker
This setup assumes the running of `attack.ipynb` and `comparison.ipynb`. Note that it is difficult to run `movie.ipynb` in a docker container due to the authentication of a camera device.

If you can use GPUs,

```console
docker-compose -f "docker/docker-compose.gpu.yaml" up -d --build 
```

If you can use only CPUs,

```console
docker-compose -f "docker/docker-compose.cpu.yaml" up -d --build 
```

## pyenv
This setup assumes the running of `movie.ipynb`. You may also be able to run `attack.ipynb` with this setup. Note that `attack.ipynb` and `comparison.ipynb` should be run in a docker container with GPUs.

```console
pyenv install 3.11.8
pyenv global 3.11.8
pip install matplotlib==3.8.3 seaborn==0.13.2 opencv-python==4.8.1.78 torch==2.1.2 torchvision==0.16.2 ipython==8.22.2 torchattacks==3.5.1
```

# Contents
`comparison.ipynb`

- Comparison with FGSM, PGD, and APGD in CIFAR-10
- Comparison with cross-entropy and DLR loss under gradient masking

`movie.ipynb`

- Simple prediction for objects in a movie. You can test physical adversarial examples such as adversarial patches.
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_22-06.html
FROM nvcr.io/nvidia/pytorch:22.06-py3

# Python 3.8.13
# CUDA 11.7 driver version 515.48.08

# apt-cache policy git -> Installed: 1:2.25.1-1ubuntu3.4
# apt-cache policy texlive-fonts-recommended -> Installed: 2019.20200218-1

# pip3 list -> (excerpt)
# ipykernel                     6.14.0
# ipython                       8.4.0
# latex                         0.7.0
# notebook                      6.4.10
# numpy                         1.22.4
# pandas                        1.3.5
# Pillow                        9.0.1
# scikit-learn                  1.1.1
# scipy                         1.6.3
# torch                         1.13.0a0+340c412
# torchvision                   0.13.0a0
# tqdm                          4.64.0

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive \
apt-get install -y \
texlive=2019.20200218-1

RUN apt-get install -y \
texlive-latex-extra=2019.202000218-1 \
dvipng=1.15-1.1 \
cm-super=0.3.4-15

RUN apt update
RUN apt install -y \
texlive-luatex=2019.20200218-1 \
texlive-lang-japanese=2019.20200218-1

RUN python3 -m pip install --upgrade pip

RUN pip3 install \
gpustat==1.0.0 \
ipywidgets==8.0.4 \
japanize-matplotlib==1.1.3 \
latex==0.7.0 \
lightning-bolts==0.5.0 \
matplotlib==3.6.2 \
patchworklib==0.5.2 \
pytorch-lightning==1.6.4 \
seaborn==0.11.2 \
torchattacks==3.2.6 \
torchmetrics==0.9.2 \
git+https://github.com/RobustBench/robustbench@ec26a6cd0b0812135270c3659caabcab80701b15

RUN echo 'export OMP_NUM_THREADS=3' >> ~/.bashrc
RUN echo "alias ls='ls --color'" >> ~/.bashrc
RUN echo "PS1='\[\e[36m\]\H \w $ \[\e[0m\]'" >> ~/.bashrc
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_22-06.html
FROM nvcr.io/nvidia/pytorch:22.06-py3

# Python 3.8.13
# PyTorch Version 1.13.0a0+340c412
# CUDA 11.7 driver version 515.48.08

# scikit-learn==0.24.2
# torchvision==0.13.0a0

RUN apt-get update
RUN apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive \
apt-get install -y \
texlive

RUN apt-get install -y \
texlive-latex-extra \
texlive-fonts-recommended \
dvipng \
cm-super

RUN apt update
RUN apt install -y \
texlive-luatex \
texlive-lang-japanese

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
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:24.02-py3

# docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.02-py3
# -> PyTorch Version 2.3.0a0+ebedce2
# -> CUDA 12.3 driver version 545.23.08 with kernel driver version 515.48.07.

# python -V -> Python 3.10.12

# apt-cache policy git -> Installed: 1:2.34.1-1ubuntu1.10

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
apt-get install -y \
texlive=2021.20220204-1 \
texlive-latex-extra=2021.20220204-1 \
texlive-luatex=2021.20220204-1 \
texlive-lang-japanese=2021.20220204-1 \
cm-super=0.3.4-17 \
dvipng=1.15-1.1

# pip3 list | grep -E 'ipykernel|ipython|matplotlib|notebook|numpy|opencv|pandas|pillow|scikit-learn|scipy|sympy|torch|tqdm'
# ->
# ipykernel                 6.29.2
# ipython                   8.21.0
# ipython-genutils          0.2.0
# matplotlib                3.8.2
# matplotlib-inline         0.1.6
# notebook                  6.4.10
# numpy                     1.24.4
# opencv                    4.7.0
# pandas                    1.5.3
# pillow                    10.2.0
# pytorch-quantization      2.1.2
# scikit-learn              1.2.0
# scipy                     1.12.0
# sympy                     1.12
# torch                     2.3.0a0+ebedce2
# torch-tensorrt            2.3.0a0
# torchdata                 0.7.1a0
# torchtext                 0.17.0a0
# torchvision               0.18.0a0
# tqdm                      4.66.1

RUN python3 -m pip install --upgrade pip
RUN pip3 install \
gpustat==1.1.1 \
ipywidgets==8.1.2 \
torchmetrics==1.3.2 \
pytorch-lightning==1.9.5 \
lightning-bolts==0.7.0 \
japanize-matplotlib==1.1.3 \
seaborn==0.13.2 \
patchworklib==0.6.3 \
latex==0.7.0 \
pypdf==4.1.0 \
torchattacks==3.5.1 \
git+https://github.com/RobustBench/robustbench@34dda3a2367b6e2d04027fbef42b7c2d6cedf4a3

# pip check

RUN echo 'export OMP_NUM_THREADS=3' >> ~/.bashrc
RUN echo "alias ls='ls --color'" >> ~/.bashrc
RUN echo "PS1='\[\e[36m\]\H \w $ \[\e[0m\]'" >> ~/.bashrc
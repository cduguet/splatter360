FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unrar \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

# Configure Conda to use conda-forge only
RUN echo "channels:\n  - conda-forge\n" > /root/.condarc

# Create environment
RUN conda create -y -n splat360 -c conda-forge --override-channels python=3.10

# Activate environment and install dependencies
SHELL ["conda", "run", "-n", "splat360", "/bin/bash", "-c"]

RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install conda packages
RUN conda install -y -c nvidia/label/cuda-11.8.0 -c conda-forge --override-channels cuda-nvcc cuda-toolkit
RUN conda install -y -c conda-forge --override-channels gcc=11 gxx=11

# Install pip requirements
COPY requirements_mod.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Additional dependencies identified during setup
RUN pip install \
    gdown \
    pytorch-lightning \
    opencv-python \
    colorama \
    dacite \
    beartype \
    e3nn \
    scikit-video \
    colorspacious \
    matplotlib \
    "numpy<2"

# Install diff-gaussian-rasterization
# Setting CUDA_HOME for compilation and ARCH list
ENV CUDA_HOME=/root/miniconda3/envs/splat360
ENV TORCH_CUDA_ARCH_LIST="8.0"
RUN pip install --no-build-isolation git+https://github.com/dcharatan/diff-gaussian-rasterization-modified

# Clone Depth-Anything-V2
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2 /root/Depth-Anything-V2

# Set working directory
WORKDIR /app

# Default command
CMD ["/bin/bash"]
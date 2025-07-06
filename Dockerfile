# Use a more widely compatible base image
FROM python:3.10-slim

# Set environment variables to disable CPU optimizations that might cause issues
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_CORETYPE=ARMV8 \
    MKL_SERVICE_FORCE_INTEL=1 \
    MKL_DISABLE_FAST_MM=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libusb-1.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch without CUDA first (more compatible)
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install common dependencies
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas \
    matplotlib \
    scikit-image \
    tensorboardX \
    torchsummary \
    tqdm \
    transforms3d \
    trimesh \
    autolab_core \
    cvxopt \
    numba \
    thop>=0.1.1.post2209072238

# Install OpenCV (often causes issues)
RUN pip install --no-cache-dir opencv-python-headless

# Install Open3D (often has compatibility issues)
RUN pip install --no-cache-dir open3d==0.17.0

# Install grasp_nms if possible
RUN pip install --no-cache-dir grasp_nms || \
    echo "Could not install grasp_nms from pip, may need manual installation"

# Set environment variable to make matplotlib work in docker
ENV PYTHONFAULTHANDLER=1 \
    MPLBACKEND=Agg

# Set the entrypoint to python3
ENTRYPOINT ["bash"]

# Default to running the demo script
CMD ["demo.sh"]

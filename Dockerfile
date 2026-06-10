FROM ubuntu:20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda

# 安装系统依赖与 Miniconda
RUN apt-get update && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh

ENV PATH=${CONDA_DIR}/bin:$PATH

#  Conda 创建环境
WORKDIR /app
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

RUN conda create -n fawa python=3.6 -y && \
    conda run -n fawa python -m pip install --upgrade pip setuptools wheel

# 复制项目文件
COPY . /app/

# 验证环境
RUN python --version

# 设置 bash 为入口点
CMD ["/bin/bash"]
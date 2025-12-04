# ROCm 6.4.4ベースのPyTorch環境
FROM rocm/dev-ubuntu-22.04:6.4.4

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_VERSION=6.4.4
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# pipのアップグレード
RUN pip3 install --upgrade pip

# PyTorch ROCm版のインストール
# ROCm 6.4.4に対応するPyTorch 2.5系をインストール
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# 作業ディレクトリの設定
WORKDIR /workspace
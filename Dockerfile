# AMD GPU用のベースイメージに変更
FROM rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# ROCm用のPyTorchを先にインストール（requirements.txt内のtorch/torchvisionを上書き）
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
# その他の依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# AMD GPU環境変数
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HIP_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
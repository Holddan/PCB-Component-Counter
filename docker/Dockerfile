# Sử dụng base image PyTorch có hỗ trợ CUDA 12.1 và CUDNN
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mặc định khi vào container sẽ mở bash
CMD ["/bin/bash"]
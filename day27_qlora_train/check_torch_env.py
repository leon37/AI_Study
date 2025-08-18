import torch

# 1. 查看是否支持 CUDA
print("CUDA available:", torch.cuda.is_available())

# 2. 当前使用的设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Current device:", device)

# 3. 如果是 CUDA，打印显卡信息
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 4. 查看 PyTorch 编译时的 CUDA 版本
print("Compiled CUDA version:", torch.version.cuda)

# 5. 查看 cuDNN 版本
print("cuDNN version:", torch.backends.cudnn.version())
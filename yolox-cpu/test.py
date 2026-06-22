import torch
print(torch.cuda.is_available())  # 应返回True
print(torch.version.cuda)         # 应显示CUDA版本（如"11.8"）
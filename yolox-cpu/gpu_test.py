import torch
print(torch.cuda.is_available())  # 应为True
print(torch.cuda.get_device_name(0))  # 应显示显卡型号
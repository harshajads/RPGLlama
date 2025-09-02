import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version reported by torch:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

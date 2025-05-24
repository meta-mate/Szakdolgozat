import torch

print(torch.__version__)
print(torch.cuda.is_available())  # Should return True if CUDA is detected
print(torch.version.cuda)  # Should output the installed CUDA version
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.2f} GB total")
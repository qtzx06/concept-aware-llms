import torch

# check if a ROCm-compatible GPU is available
is_available = torch.cuda.is_available()

print(f"ROCm-enabled GPU available: {is_available}")

# if a GPU is available, print its name
if is_available:
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} ROCm device(s).")
    
    # get the name of the first GPU (device 0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device 0 Name: {gpu_name}")

    # Example: create a tensor and move it to the GPU
    print("\nTesting tensor operations on GPU...")
    try:
        # create a tensor on the CPU
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        print(f"Tensor on CPU: {cpu_tensor}")

        # move the tensor to the GPU (device 0)
        gpu_tensor = cpu_tensor.to('cuda:0')
        print(f"Tensor on GPU: {gpu_tensor}")
        print("✅ Test successful! PyTorch is using the ROCm device.")
    except Exception as e:
        print(f"❌ An error occurred during tensor operations: {e}")

else:
    print("❌ Test failed. PyTorch cannot find a ROCm-enabled GPU.")
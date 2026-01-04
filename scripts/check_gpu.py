"""
Check GPU Availability and Setup
"""
import torch
import sys

def check_gpu():
    """Check GPU availability and print information"""
    
    print("🖥️  GPU Setup Check")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✅ GPU Detected!")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        
        # Memory info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            print(f"   Memory Allocated: {memory_allocated:.2f} GB")
            print(f"   Memory Reserved: {memory_reserved:.2f} GB")
    else:
        print("❌ No GPU detected!")
        print("\n💡 Solutions:")
        print("   1. Use Google Colab (free GPU):")
        print("      - Go to: https://colab.research.google.com/")
        print("      - Runtime → Change runtime type → GPU")
        print("   2. Use Kaggle Notebooks (free GPU):")
        print("      - Go to: https://www.kaggle.com/code")
        print("      - Settings → Accelerator → GPU")
        print("   3. Install CUDA locally (if you have NVIDIA GPU):")
        print("      - Download from: https://developer.nvidia.com/cuda-downloads")
    
    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # Test tensor operations
    print("\n🧪 Testing Tensor Operations:")
    try:
        if cuda_available:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("   ✅ GPU tensor operations working!")
        else:
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            z = torch.matmul(x, y)
            print("   ✅ CPU tensor operations working (but slow)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    
    return cuda_available

if __name__ == "__main__":
    has_gpu = check_gpu()
    sys.exit(0 if has_gpu else 1)


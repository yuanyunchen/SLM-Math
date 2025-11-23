"""
Check if vLLM is available and provide installation guidance
"""

def check_vllm_available() -> bool:
    """Check if vLLM is installed"""
    try:
        import vllm
        return True
    except ImportError:
        return False


def print_vllm_info():
    """Print vLLM installation and usage information"""
    available = check_vllm_available()
    
    if available:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
        print("vLLM is available for accelerated inference.")
    else:
        print("vLLM is NOT installed.")
        print("\nTo use vLLM for faster inference, install it with:")
        print("  pip install vllm")
        print("\nNote:")
        print("  - vLLM requires CUDA (GPU)")
        print("  - Recommended for A100, H100 GPUs")
        print("  - May require specific PyTorch/CUDA versions")
        print("\nFor more information: https://docs.vllm.ai/")


if __name__ == "__main__":
    print_vllm_info()


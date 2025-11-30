"""
Utility to check vLLM availability and compatibility
"""

import sys


def check_vllm_available(verbose: bool = False) -> bool:
    """
    Check if vLLM is installed and available
    
    Args:
        verbose: If True, print detailed error messages
    
    Returns:
        True if vLLM is available, False otherwise
    """
    try:
        import vllm
        
        # Check for required CUDA support
        try:
            import torch
            if not torch.cuda.is_available():
                if verbose:
                    print("WARNING: vLLM requires CUDA, but CUDA is not available")
                return False
        except ImportError:
            if verbose:
                print("WARNING: PyTorch not found, cannot verify CUDA availability")
            return False
        
        if verbose:
            print(f"vLLM version {vllm.__version__} is available")
        
        return True
        
    except ImportError as e:
        if verbose:
            print(f"vLLM is not installed: {e}")
            print("Install with: pip install vllm")
        return False
    
    except Exception as e:
        if verbose:
            print(f"Error checking vLLM availability: {e}")
        return False


def get_vllm_info() -> dict:
    """
    Get detailed information about vLLM installation
    
    Returns:
        Dictionary with vLLM information, or None if not available
    """
    if not check_vllm_available():
        return None
    
    try:
        import vllm
        import torch
        
        info = {
            'vllm_version': vllm.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    """Test vLLM availability"""
    print("Checking vLLM availability...")
    print("="*80)
    
    is_available = check_vllm_available(verbose=True)
    
    if is_available:
        print("\n✓ vLLM is available and ready to use")
        print("\nDetailed Information:")
        print("-"*80)
        info = get_vllm_info()
        if info:
            for key, value in info.items():
                print(f"  {key}: {value}")
    else:
        print("\n✗ vLLM is not available")
        print("\nTo install vLLM, run:")
        print("  pip install vllm")
        print("\nNote: vLLM requires:")
        print("  - CUDA 11.8 or higher")
        print("  - Compatible NVIDIA GPU")
        print("  - Linux operating system")
    
    print("="*80)
    
    sys.exit(0 if is_available else 1)


















#!/usr/bin/env python3
"""
环境检查脚本

检查服务器环境是否满足项目运行要求
"""

import sys
import subprocess
from typing import Tuple

def check_python_version() -> bool:
    """检查 Python 版本"""
    print(f"Python version: {sys.version}")
    version = sys.version_info
    
    if version.major == 3 and version.minor == 10:
        print("✅ Python 3.10 detected (Recommended)")
        return True
    elif version.major == 3 and version.minor == 11:
        print("⚠️  Python 3.11 detected (Compatible but not optimal)")
        return True
    elif version.major == 3 and version.minor == 9:
        print("⚠️  Python 3.9 detected (May have compatibility issues)")
        return False
    elif version.major == 3 and version.minor >= 12:
        print("❌ Python 3.12+ not recommended (BitsAndBytes compatibility issues)")
        return False
    else:
        print(f"❌ Python {version.major}.{version.minor} not supported")
        return False

def check_import(module_name: str, package_name: str = None, show_version: bool = True) -> bool:
    """检查模块是否安装"""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        version_str = f" (v{version})" if show_version and version != 'unknown' else ""
        print(f"✅ {package_name or module_name}{version_str}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} NOT installed")
        return False

def check_cuda() -> Tuple[bool, dict]:
    """检查 CUDA 环境"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available (CPU mode only)")
            return False, {}
        
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        
        info = {
            'cuda_version': cuda_version,
            'device_count': device_count,
            'devices': []
        }
        
        print(f"✅ CUDA available: {cuda_version}")
        print(f"   Device count: {device_count}")
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            
            info['devices'].append({
                'name': name,
                'memory_gb': memory_gb
            })
            
            print(f"   GPU {i}: {name}")
            print(f"           Memory: {memory_gb:.1f} GB")
            
            # 显存要求检查
            if memory_gb < 12:
                print(f"           ⚠️  Warning: <12GB, may fail for LLM training")
            elif memory_gb < 24:
                print(f"           ⚠️  Warning: <24GB, recommend reducing batch_size")
            else:
                print(f"           ✅ Sufficient memory")
        
        return True, info
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False, {}
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False, {}

def check_versions() -> dict:
    """检查关键库的版本号"""
    versions = {}
    
    try:
        import torch
        versions['torch'] = torch.__version__
        
        # 检查是否满足最低要求
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version >= (2, 2):
            print(f"   PyTorch {torch.__version__} ✅")
        else:
            print(f"   PyTorch {torch.__version__} ⚠️  (Recommend ≥2.2.0)")
    except:
        pass
    
    try:
        import transformers
        versions['transformers'] = transformers.__version__
        
        # 检查是否满足最低要求
        version_parts = transformers.__version__.split('.')
        major_minor = tuple(map(int, version_parts[:2]))
        if major_minor >= (4, 40):
            print(f"   Transformers {transformers.__version__} ✅")
        else:
            print(f"   Transformers {transformers.__version__} ⚠️  (Recommend ≥4.40.0)")
    except:
        pass
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
        
        version_parts = numpy.__version__.split('.')
        major_minor = tuple(map(int, version_parts[:2]))
        if major_minor >= (1, 26):
            print(f"   NumPy {numpy.__version__} ✅")
        else:
            print(f"   NumPy {numpy.__version__} ⚠️  (Recommend ≥1.26.0)")
    except:
        pass
    
    try:
        import bitsandbytes
        versions['bitsandbytes'] = bitsandbytes.__version__
        print(f"   BitsAndBytes {bitsandbytes.__version__} ✅")
    except:
        pass
    
    return versions

def main():
    print("="*70)
    print(" 🔍 POI Recommendation System - Environment Check")
    print("="*70)
    print()
    
    all_pass = True
    
    # 1. Python 版本
    print("📌 Python Version")
    print("-" * 70)
    if not check_python_version():
        all_pass = False
    print()
    
    # 2. 核心依赖
    print("📌 Core Dependencies")
    print("-" * 70)
    core_deps = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('trl', 'TRL'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('accelerate', 'Accelerate'),
    ]
    
    for module, name in core_deps:
        if not check_import(module, name):
            all_pass = False
    print()
    
    # 3. 数据处理库
    print("📌 Data Processing Libraries")
    print("-" * 70)
    data_deps = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('tqdm', 'tqdm'),
    ]
    
    for module, name in data_deps:
        if not check_import(module, name):
            all_pass = False
    print()
    
    # 4. 可视化库
    print("📌 Visualization Libraries")
    print("-" * 70)
    vis_deps = [
        ('matplotlib', 'Matplotlib'),
        ('tensorboard', 'TensorBoard'),
    ]
    
    for module, name in vis_deps:
        check_import(module, name)  # 不是必需的
    print()
    
    # 5. 可选库
    print("📌 Optional Libraries")
    print("-" * 70)
    optional_deps = [
        ('openlocationcode', 'OpenLocationCode'),
        ('einops', 'Einops'),
        ('wandb', 'Weights & Biases'),
    ]
    
    for module, name in optional_deps:
        check_import(module, name)
    print()
    
    # 6. 版本检查
    print("📌 Version Requirements")
    print("-" * 70)
    check_versions()
    print()
    
    # 7. CUDA 检查
    print("📌 GPU/CUDA Environment")
    print("-" * 70)
    cuda_ok, cuda_info = check_cuda()
    print()
    
    # 8. 建议
    print("="*70)
    print(" 📋 Summary")
    print("="*70)
    
    if all_pass and cuda_ok:
        print("✅ Environment check PASSED!")
        print("   All required dependencies are installed and configured correctly.")
        
        if cuda_info and cuda_info.get('devices'):
            for i, device in enumerate(cuda_info['devices']):
                memory = device['memory_gb']
                if memory >= 24:
                    print(f"   GPU {i}: Ready for full LLM training")
                elif memory >= 12:
                    print(f"   GPU {i}: Can run LLM with reduced batch_size")
                else:
                    print(f"   GPU {i}: Only suitable for Semantic ID training")
    else:
        print("⚠️  Environment check FAILED!")
        print()
        print("   Missing dependencies. Install them with:")
        print("   pip install -r requirements.txt")
        print()
        
        if not cuda_ok:
            print("   CUDA not available. For GPU training:")
            print("   1. Install NVIDIA driver")
            print("   2. Install PyTorch with CUDA:")
            print("      conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
    
    print("="*70)
    
    # 返回退出码
    sys.exit(0 if all_pass else 1)

if __name__ == '__main__':
    main()

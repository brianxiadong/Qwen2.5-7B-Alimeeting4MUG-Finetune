#!/usr/bin/env python3
"""
Flash Attention 环境检测脚本
用于确定正确的 Flash Attention wheel 版本

使用方法:
    python scripts/check_flash_attn_env.py

然后去 https://github.com/Dao-AILab/flash-attention/releases 下载对应版本
"""

import sys
import platform

def get_cxx11_abi():
    """检测 CXX11 ABI 设置"""
    try:
        import torch
        return torch._C._GLIBCXX_USE_CXX11_ABI
    except:
        return None

def main():
    print("=" * 60)
    print("🔍 Flash Attention 环境检测")
    print("=" * 60)
    
    # Python 版本
    v = sys.version_info
    python_tag = f"cp{v.major}{v.minor}"
    print(f"\n📌 Python 版本: {v.major}.{v.minor}.{v.micro}")
    print(f"   wheel 标签: {python_tag}")
    
    # PyTorch 版本
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]
        # 提取主版本号 (如 2.5.0 -> 2.5)
        torch_major_minor = '.'.join(torch_version.split('.')[:2])
        print(f"\n📌 PyTorch 版本: {torch.__version__}")
        print(f"   wheel 标签: torch{torch_major_minor}")
    except ImportError:
        print("\n❌ PyTorch 未安装")
        torch_major_minor = None
        return
    
    # CUDA 版本
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cuda_major = cuda_version.split('.')[0]  # 如 12.1 -> 12
        print(f"\n📌 CUDA 版本: {cuda_version}")
        print(f"   wheel 标签: cu{cuda_major}")
    else:
        print("\n❌ CUDA 不可用")
        cuda_major = None
        return
    
    # CXX11 ABI
    cxx11_abi = get_cxx11_abi()
    if cxx11_abi is not None:
        abi_tag = "TRUE" if cxx11_abi else "FALSE"
        print(f"\n📌 CXX11 ABI: {cxx11_abi}")
        print(f"   wheel 标签: cxx11abi{abi_tag}")
    else:
        abi_tag = "FALSE"  # 默认
        print(f"\n⚠️ CXX11 ABI: 未知 (默认使用 FALSE)")
    
    # 系统架构
    arch = platform.machine()
    os_name = platform.system().lower()
    print(f"\n📌 系统架构: {os_name}_{arch}")
    
    # 生成推荐的 wheel 文件名
    print("\n" + "=" * 60)
    print("📦 推荐的 Flash Attention wheel 文件名:")
    print("=" * 60)
    
    # Flash Attention wheel 命名格式:
    # flash_attn-{version}+cu{cuda}torch{torch}cxx11abi{ABI}-{python}-{python}-{platform}.whl
    wheel_name = f"flash_attn-2.8.3+cu{cuda_major}torch{torch_major_minor}cxx11abi{abi_tag}-{python_tag}-{python_tag}-{os_name}_{arch}.whl"
    
    print(f"\n  {wheel_name}")
    
    print("\n" + "=" * 60)
    print("🔗 下载链接:")
    print("=" * 60)
    print("\n  https://github.com/Dao-AILab/flash-attention/releases")
    print(f"\n  搜索关键词: cu{cuda_major} torch{torch_major_minor} {python_tag}")
    
    # 快速安装命令
    print("\n" + "=" * 60)
    print("📋 安装命令 (下载后运行):")
    print("=" * 60)
    print(f"\n  pip install {wheel_name}")
    
    # 一行检测命令（方便复制）
    print("\n" + "=" * 60)
    print("📋 快速一行检测命令:")
    print("=" * 60)
    print('''
  python -c "import torch; import sys; v=sys.version_info; print(f'Python: cp{v.major}{v.minor}, PyTorch: {torch.__version__.split(\\"+\\")[0]}, CUDA: {torch.version.cuda}, CXX11_ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"
''')

if __name__ == "__main__":
    main()

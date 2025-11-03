#!/usr/bin/env python3
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

"""
LFM Installation & Setup Script
===============================
Automated setup for the Lattice Field Medium framework on all platforms.
Detects your system and installs everything needed to run LFM.
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def print_header():
    """Print setup banner"""
    print("=" * 60)
    print("  LFM SETUP - Lattice Field Medium Installation")
    print("  Unified Physics Framework Installer")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ ERROR: Python 3.9+ required")
        print("   Please upgrade Python: https://python.org/downloads/")
        return False
    
    print("✅ Python version compatible")
    return True

def check_tkinter():
    """Check if tkinter is available for GUI"""
    try:
        import tkinter as tk
        print("✅ Tkinter available - GUI interface will work")
        return True
    except ImportError:
        print("⚠️  Tkinter not available - Console interface only")
        print("   On Linux: sudo apt-get install python3-tk")
        print("   On macOS: Tkinter should be included with Python")
        return False

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python dependencies...")
    
    # Core dependencies
    packages = [
        "numpy>=1.24.0",
        "matplotlib>=3.7.0", 
        "scipy>=1.10.0",
        "h5py>=3.8.0",
        "pytest>=7.3.0"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {package}")
            return False
    
    return True

def check_gpu_support():
    """Check for NVIDIA GPU and offer CuPy installation"""
    print("\n🚀 Checking GPU support...")
    
    try:
        # Try to detect NVIDIA GPU
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
            print(f"   🎮 NVIDIA GPU detected: {gpu_name}")
            
            # Ask user if they want GPU acceleration
            while True:
                response = input("   Install GPU acceleration (CuPy)? [y/N]: ").lower()
                if response in ['y', 'yes']:
                    print("   Installing CuPy for GPU acceleration...")
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", "cupy-cuda12x"
                        ], stdout=subprocess.DEVNULL)
                        print("   ✅ GPU acceleration installed")
                        return True
                    except subprocess.CalledProcessError:
                        print("   ❌ GPU installation failed - continuing with CPU")
                        return False
                elif response in ['n', 'no', '']:
                    print("   ⏭️  Skipping GPU acceleration")
                    return False
                else:
                    print("   Please enter 'y' or 'n'")
        else:
            print("   💻 No NVIDIA GPU detected - using CPU only")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   💻 Unable to check GPU - using CPU only")
        return False

def run_quick_test():
    """Run a quick validation test"""
    print("\n🧪 Running quick validation test...")
    
    try:
        # Import core LFM modules to test installation
        sys.path.append(str(Path(__file__).parent))
        
        print("   Testing core imports...")
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import scipy
        import h5py
        import pytest
        
        # Test LFM equation module
        print("   Testing LFM equation solver...")
        from lfm_equation import LFMEquation
        
        # Quick 1D test
        equation = LFMEquation(
            nx=32, dx=0.1, dt=0.01, 
            alpha=1.0, beta=1.0, chi=0.1,
            boundary='periodic', use_gpu=False
        )
        
        # Run 10 steps
        for _ in range(10):
            equation.step()
        
        print("   ✅ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def create_desktop_shortcuts():
    """Create shortcuts for easy access"""
    print("\n🔗 Creating convenient shortcuts...")
    
    # Get current directory
    lfm_dir = Path(__file__).parent
    
    # Windows shortcuts
    if platform.system() == "Windows":
        try:
            # Try to create Windows shortcuts using basic approach
            print("   ⏭️  Advanced Windows shortcuts require pywin32 package")
            print("   💡 Use batch files instead: run_lfm_console.bat, run_lfm_gui.bat")
            
        except Exception as e:
            print(f"   ⏭️  Skipping Windows shortcuts: {e}")
    
    # Create simple batch/shell scripts for all platforms
    if platform.system() == "Windows":
        # Batch files
        with open(lfm_dir / "run_lfm_console.bat", "w") as f:
            f.write(f'@echo off\ncd /d "{lfm_dir}"\npython lfm_control_center.py\npause\n')
        
        with open(lfm_dir / "run_lfm_gui.bat", "w") as f:
            f.write(f'@echo off\ncd /d "{lfm_dir}"\npython lfm_gui.py\n')
            
        print("   ✅ Batch files created (run_lfm_console.bat, run_lfm_gui.bat)")
    
    else:
        # Shell scripts for macOS/Linux
        with open(lfm_dir / "run_lfm_console.sh", "w") as f:
            f.write(f'#!/bin/bash\ncd "{lfm_dir}"\npython3 lfm_control_center.py\n')
        os.chmod(lfm_dir / "run_lfm_console.sh", 0o755)
        
        with open(lfm_dir / "run_lfm_gui.sh", "w") as f:
            f.write(f'#!/bin/bash\ncd "{lfm_dir}"\npython3 lfm_gui.py\n')
        os.chmod(lfm_dir / "run_lfm_gui.sh", 0o755)
        
        print("   ✅ Shell scripts created (run_lfm_console.sh, run_lfm_gui.sh)")

def print_usage_instructions():
    """Print how to use LFM after installation"""
    print("\n" + "=" * 60)
    print("  🎉 LFM INSTALLATION COMPLETE!")
    print("=" * 60)
    print()
    print("🚀 HOW TO RUN LFM:")
    print()
    
    if platform.system() == "Windows":
        print("   Option 1: Double-click desktop shortcuts")
        print("   Option 2: Double-click batch files:")
        print("            • run_lfm_console.bat (text interface)")
        print("            • run_lfm_gui.bat (Windows interface)")
        print("   Option 3: Command line:")
        print("            python lfm_control_center.py")
        print("            python lfm_gui.py")
    else:
        print("   Option 1: Run shell scripts:")
        print("            ./run_lfm_console.sh (text interface)")
        print("            ./run_lfm_gui.sh (graphical interface)")
        print("   Option 2: Command line:")
        print("            python3 lfm_control_center.py")
        print("            python3 lfm_gui.py")
    
    print()
    print("📋 FIRST STEPS:")
    print("   1. Run a quick test: REL-01 (Relativistic propagation)")
    print("   2. View results in: results/Relativistic/REL-01/")
    print("   3. Try the emergence validation demo")
    print("   4. Explore the full test suite (55 tests total)")
    print()
    print("📚 DOCUMENTATION:")
    print("   • README.md - Complete overview")
    print("   • docs/ - Detailed documentation")
    print("   • requirements.txt - Dependency list")
    print()
    print("💬 SUPPORT:")
    print("   Email: latticefieldmediumresearch@gmail.com")
    print("   ORCID: https://orcid.org/0009-0004-0327-6528")
    print()

def main():
    """Main installation process"""
    print_header()
    
    # System info
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check tkinter
    has_gui = check_tkinter()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        sys.exit(1)
    
    print("✅ All dependencies installed successfully!")
    
    # Check GPU support
    check_gpu_support()
    
    # Run validation test
    if not run_quick_test():
        print("\n⚠️  Installation may have issues - check error messages above")
    
    # Create shortcuts
    create_desktop_shortcuts()
    
    # Final instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
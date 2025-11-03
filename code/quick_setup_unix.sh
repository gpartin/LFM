#!/bin/bash
# Copyright (c) 2025 Greg D. Partin. All rights reserved.
# Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
# See LICENSE file in project root for full license text.
# Commercial use prohibited without explicit written permission.
# Contact: latticefieldmediumresearch@gmail.com

echo
echo "===================================================="
echo "  LFM Quick Setup - Lattice Field Medium"
echo "===================================================="
echo
echo "This will install LFM on your macOS/Linux system."
echo

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.9+ first."
    exit 1
fi

echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy>=1.24.0 matplotlib>=3.7.0 scipy>=1.10.0 h5py>=3.8.0 pytest>=7.3.0

echo
echo "Testing installation..."
python3 -c "import numpy, matplotlib, scipy, h5py, pytest; print('✅ All dependencies installed!')"

echo
echo "Creating shortcuts..."
cat > run_lfm_console.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 lfm_control_center.py
EOF

cat > run_lfm_gui.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 lfm_gui.py
EOF

chmod +x run_lfm_console.sh
chmod +x run_lfm_gui.sh

echo
echo "===================================================="
echo "  🎉 LFM Installation Complete!"
echo "===================================================="
echo
echo "To start LFM:"
echo "  • Run: ./run_lfm_gui.sh (graphical interface)"
echo "  • Run: ./run_lfm_console.sh (text interface)"
echo "  • Or run: python3 lfm_gui.py"
echo
echo "First test: Try REL-01 (takes ~5 seconds)"
echo
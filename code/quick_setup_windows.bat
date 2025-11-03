@echo off
REM Copyright (c) 2025 Greg D. Partin. All rights reserved.
REM Licensed under CC BY-NC-ND 4.0 (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International).
REM See LICENSE file in project root for full license text.
REM Commercial use prohibited without explicit written permission.
REM Contact: latticefieldmediumresearch@gmail.com

echo.
echo ====================================================
echo   LFM Quick Setup - Lattice Field Medium
echo ====================================================
echo.
echo This will install LFM on your Windows system.
echo.
pause

echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install numpy>=1.24.0 matplotlib>=3.7.0 scipy>=1.10.0 h5py>=3.8.0 pytest>=7.3.0

echo.
echo Testing installation...
python -c "import numpy, matplotlib, scipy, h5py, pytest; print('✅ All dependencies installed!')"

echo.
echo Creating shortcuts...
echo @echo off > run_lfm_console.bat
echo cd /d "%~dp0" >> run_lfm_console.bat
echo python lfm_control_center.py >> run_lfm_console.bat
echo pause >> run_lfm_console.bat

echo @echo off > run_lfm_gui.bat
echo cd /d "%~dp0" >> run_lfm_gui.bat
echo python lfm_gui.py >> run_lfm_gui.bat

echo.
echo ====================================================
echo   🎉 LFM Installation Complete!
echo ====================================================
echo.
echo To start LFM:
echo   • Double-click: run_lfm_gui.bat (Windows interface)
echo   • Double-click: run_lfm_console.bat (text interface)
echo   • Or run: python lfm_gui.py
echo.
echo First test: Try REL-01 (takes ~5 seconds)
echo.
pause
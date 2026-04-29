@echo off
chcp 65001
cls
echo ============================================
echo    AlphaMCTS 可视化 GUI 系统
echo ============================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请安装Python 3.7+
    pause
    exit /b 1
)

REM 安装依赖
echo [1/2] 检查依赖...
pip install PyQt5 -q
if errorlevel 1 (
    echo [警告] 依赖安装可能失败，尝试继续...
)

REM 运行GUI
echo [2/2] 启动GUI系统...
echo.
python mcts_gui.py

pause

@echo off
chcp 65001 >nul
title YoloSharpNet48Demo 重建并运行验证
echo ============================================
echo  目录: %~dp0
echo  操作: 完整重建 --no-incremental 并运行
echo ============================================
cd /d "%~dp0"
echo.
echo [1/2] dotnet build ...
dotnet build -c Debug --no-incremental
if errorlevel 1 (
    echo.
    echo !!! 编译失败，请把上方错误信息发回 !!!
    pause
    exit /b 1
)
echo.
echo [2/2] 运行 YoloSharpNet48Demo.exe ...
echo.
bin\Debug\net48\YoloSharpNet48Demo.exe
echo.
echo 退出码: %errorlevel%  （0=成功）
pause

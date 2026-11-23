@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ==============================================================
REM  推送到 GitHub
REM
REM  用法:  push_to_github.bat
REM
REM  仓库: https://github.com/MokingMid/strawberry-maturity-demo
REM
REM  前置条件：
REM    1) 已在 https://github.com/new 建好空仓库 strawberry-maturity-demo
REM       （不要勾选 README / .gitignore / LICENSE，否则会冲突）
REM    2) 仓库设为 Public
REM
REM  认证：Git Credential Manager (GCM) 会自动弹出浏览器让你登录 GitHub，
REM        不用手动粘贴 Token。
REM ==============================================================

set "REPO_DIR=%~dp0"
set "GITHUB_USER=MokingMid"
set "REPO_NAME=strawberry-maturity-demo"
set "REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git"

cd /d "%REPO_DIR%"

echo.
echo ==============================================================
echo   仓库: %REPO_URL%
echo   Pages: https://%GITHUB_USER%.github.io/%REPO_NAME%/
echo ==============================================================
echo.

REM ---------- 1. remote ----------
git remote remove origin >nul 2>&1
git remote add origin "%REPO_URL%"
git branch -M main
echo [1/3] remote 已设置

REM ---------- 2. 推送 ----------
echo [2/3] 正在推送 (约 53MB, 首次可能 2-5 分钟)
echo       ^>^> 浏览器会弹出 GitHub 登录页, 登录授权后自动继续
echo.
git push -u origin main

if errorlevel 1 (
    echo.
    echo ==============================================================
    echo   推送失败。排查顺序：
    echo.
    echo   1) 仓库还没建？先访问 https://github.com/new
    echo      仓库名填 %REPO_NAME%, 选 Public, 不要勾任何初始化文件
    echo.
    echo   2) 认证失败？可用 PAT 方式推送：
    echo      https://github.com/settings/tokens  (classic, 勾 public_repo)
    echo      然后在同目录执行：
    echo        git push -u https://%GITHUB_USER%:PAT_TOKEN@github.com/%GITHUB_USER%/%REPO_NAME%.git main
    echo.
    echo   3) 网络问题：检查是否能打开 https://github.com
    echo ==============================================================
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] 推送完成
echo.
echo ==============================================================
echo   接下来两步：
echo.
echo   1) 开启 GitHub Pages ^(静态展示站, 约 2 分钟自动发布^)
echo      https://github.com/%GITHUB_USER%/%REPO_NAME%/settings/pages
echo      Source 选 "GitHub Actions", 保存
echo      访问: https://%GITHUB_USER%.github.io/%REPO_NAME%/
echo.
echo   2) 部署实时推理网站 ^(Flask, 可上传图片跑推理^)
echo      https://render.com/deploy   ->  选本仓库 -> Apply
echo      访问: https://strawberry-maturity-demo.onrender.com
echo ==============================================================
echo.
pause
exit /b 0

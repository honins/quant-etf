#!/bin/bash

# 检查是否已安装 Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found. Please install it first."
    exit 1
fi

# 确保 venv 模块可用 (Debian/Ubuntu 需要 python3-venv)
# 这一步作为提示，如果创建失败则提示用户
python3 -m venv venv || { echo "❌ Failed to create venv. Try: apt install python3-venv"; exit 1; }

# 激活虚拟环境
source venv/bin/activate

# 创建日志目录
mkdir -p logs

# 杀死旧进程 (如果存在)
pkill -f "scheduler.py"

# 安装依赖 (使用虚拟环境的 pip)
echo "📦 Installing dependencies in venv..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 后台启动 (使用虚拟环境的 python)
echo "🚀 Starting Quant-ETF Scheduler..."
nohup python scheduler.py > logs/scheduler.log 2>&1 &

echo "✅ Started! PID: $!"
echo "📜 Check logs: tail -f logs/scheduler.log"

#!/bin/bash

# 检查是否已安装 Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found. Please install it first."
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 杀死旧进程 (如果存在)
pkill -f "python3 scheduler.py"

# 安装依赖
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 后台启动
echo "🚀 Starting Quant-ETF Scheduler..."
nohup python3 scheduler.py > logs/scheduler.log 2>&1 &

echo "✅ Started! PID: $!"
echo "📜 Check logs: tail -f logs/scheduler.log"

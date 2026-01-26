#!/bin/bash

# 停止旧容器
echo "Stopping old container..."
docker stop quant-etf-bot
docker rm quant-etf-bot

# 构建镜像
echo "Building image..."
docker build -t quant-etf:latest .

# 运行新容器
echo "Starting container..."
# -d: 后台运行
# --restart always: 开机自启/崩溃重启
# -v: 挂载数据卷，确保重启后数据不丢失
docker run -d \
  --name quant-etf-bot \
  --restart always \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env \
  quant-etf:latest

echo "✅ Deployment successful!"
echo "📜 View logs with: docker logs -f quant-etf-bot"

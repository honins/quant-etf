# ☁️ 服务器部署指南 (Server Deployment)

本指南将帮助您将 Quant-ETF 系统部署到 Linux 云服务器（如阿里云、腾讯云、AWS），实现 7x24 小时无人值守运行。

---

## 📋 准备工作

1.  **购买服务器**: 最低配置即可 (1核 2G 内存，Ubuntu 20.04/22.04)。
2.  **安装 Docker**:
    ```bash
    curl -fsSL https://get.docker.com | bash
    ```

---

## 🚀 部署步骤 (Docker 方式 - 推荐)

这是最简单、最稳定的方式。

### 1. 上传代码
将整个项目文件夹上传到服务器（可以使用 Git 或 SCP）。
```bash
# 在服务器上
git clone https://github.com/your-repo/quant-etf.git
cd quant-etf
```

### 2. 配置 .env
确保服务器上有 `.env` 文件，并填好了 Token 和 Webhook。
```bash
cp .env.example .env
vim .env
# 填入您的 Tushare Token 和 Feishu Webhook
```

### 3. 一键启动
运行我们为您准备的启动脚本：
```bash
chmod +x start.sh
./start.sh
```

### 4. 验证状态
查看运行日志，确认 scheduler 已经启动：
```bash
docker logs -f quant-etf-bot
```
如果看到 `⏳ Scheduler started...`，恭喜您，部署成功！

---

## 🛠️ 常用维护命令

*   **查看日志**: `docker logs --tail 100 -f quant-etf-bot`
*   **停止服务**: `docker stop quant-etf-bot`
*   **重启服务**: `docker restart quant-etf-bot`
*   **手动运行一次 (测试)**:
    ```bash
    docker exec -it quant-etf-bot python main.py
    ```
*   **更新代码**:
    ```bash
    git pull
    ./start.sh  # 重新构建并重启
    ```

---

## 📂 数据持久化
容器启动时挂载了以下目录，确保重启容器后数据不会丢失：
*   `/data`: 历史行情数据
*   `/reports`: 生成的日报
*   `/config`: 配置文件（包括持仓信息）
*   `/logs`: 运行日志

---

## ⏰ 定时任务说明
系统内置了 `scheduler.py`，会在容器内自动执行：
*   **09:00**: 盘前运行
*   **17:00**: 盘后运行

无需在服务器上配置 crontab。

# 使用官方 Python 轻量镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置时区为上海 (这对金融数据很重要!)
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo 'Asia/Shanghai' >/etc/timezone

# 复制依赖文件
COPY requirements.txt .

# 安装依赖 (使用清华源加速)
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tushare.pro/simple/

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p data reports logs

# 声明环境变量 (可选，运行时传入更安全)
# ENV TUSHARE_TOKEN=your_token

# 启动命令
CMD ["python", "scheduler.py"]

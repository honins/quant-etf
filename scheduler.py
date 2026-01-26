import schedule
import time
import subprocess
import sys
from datetime import datetime

def job():
    print(f"⏰ Scheduled job started at {datetime.now()}")
    # 使用 subprocess 调用 main.py，确保环境隔离
    try:
        # 使用当前的 python 解释器
        python_exe = sys.executable
        result = subprocess.run([python_exe, "main.py"], capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
    except Exception as e:
        print(f"Job failed: {e}")
    print(f"✅ Job finished at {datetime.now()}")

def main():
    print("⏳ Scheduler started. Waiting for 09:00 and 17:00...")
    
    # 设定每天 09:00 (盘前检查) 和 17:00 (盘后复盘)
    schedule.every().day.at("09:00").do(job)
    schedule.every().day.at("17:00").do(job)
    
    # 立即运行一次以验证配置 (可选，调试用)
    # job()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()

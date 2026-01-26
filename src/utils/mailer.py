import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
import markdown
from datetime import datetime

class Mailer:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "465"))
        self.sender_email = os.getenv("SENDER_EMAIL", "")
        self.sender_password = os.getenv("SENDER_PASSWORD", "") # 授权码
        self.receiver_email = os.getenv("RECEIVER_EMAIL", "hhoins@gmail.com")

    def send_report(self, report_path):
        if not self.sender_email or not self.sender_password:
            print("⚠️ Email config missing. Skipping email.")
            return

        if not os.path.exists(report_path):
            print(f"⚠️ Report file not found: {report_path}")
            return

        try:
            # 读取 Markdown 内容
            with open(report_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # 转为 HTML
            html_content = markdown.markdown(md_content, extensions=['tables'])
            
            # 美化 HTML (简单的 CSS)
            css_style = """
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                h2 { color: #34495e; margin-top: 20px; }
                h3 { color: #7f8c8d; }
                table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                strong { color: #e74c3c; }
            </style>
            """
            full_html = f"<html><head>{css_style}</head><body>{html_content}</body></html>"

            # 构建邮件
            msg = MIMEMultipart()
            today = datetime.now().strftime("%Y-%m-%d")
            msg['Subject'] = Header(f"📈 Quant-ETF Daily Report ({today})", 'utf-8')
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email

            msg.attach(MIMEText(full_html, 'html', 'utf-8'))

            # 发送
            server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, [self.receiver_email], msg.as_string())
            server.quit()
            
            print(f"✅ Email sent to {self.receiver_email}")

        except Exception as e:
            print(f"❌ Failed to send email: {e}")

import requests
import json
import os
import re

class FeishuBot:
    def __init__(self):
        self.webhook = os.getenv("FEISHU_WEBHOOK", "")

    def send_markdown(self, title, content_md):
        """
        发送 Markdown 消息 (使用卡片格式以支持更好的渲染)
        """
        if not self.webhook:
            print("⚠️ Feishu webhook not configured. Skip.")
            return

        # 飞书卡片 Markdown 不支持表格，需要简单处理
        # 策略：将表格内容转换为代码块，或者简化的文本列表
        # 这里采用简单的正则替换：将表格行放入代码块中
        formatted_content = self._optimize_markdown_for_feishu(content_md)

        # 构建卡片消息
        card = {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": formatted_content
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": "来自 Quant-ETF 自动交易系统"
                        }
                    ]
                }
            ]
        }

        payload = {
            "msg_type": "interactive",
            "card": card
        }

        try:
            resp = requests.post(self.webhook, json=payload)
            resp.raise_for_status()
            res_json = resp.json()
            if res_json.get("code") == 0:
                print("✅ Feishu notification sent.")
            else:
                print(f"❌ Feishu send failed: {res_json}")
        except Exception as e:
            print(f"❌ Feishu connection error: {e}")

    def _optimize_markdown_for_feishu(self, md_text):
        """
        优化 Markdown 文本以适应飞书卡片
        主要处理表格，将其转换为代码块，避免乱码
        """
        lines = md_text.split('\n')
        new_lines = []
        in_table = False
        
        for line in lines:
            # 简单的表格检测：以 | 开头和结尾
            if line.strip().startswith('|') and line.strip().endswith('|'):
                if not in_table:
                    new_lines.append("```text") # 开始代码块
                    in_table = True
                new_lines.append(line)
            else:
                if in_table:
                    new_lines.append("```") # 结束代码块
                    in_table = False
                new_lines.append(line)
        
        if in_table:
            new_lines.append("```")
            
        return "\n".join(new_lines)

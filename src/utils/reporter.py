import os
from datetime import datetime
from pathlib import Path
from config.settings import settings

class Reporter:
    def __init__(self):
        self.report_dir = settings.REPORTS_DIR

    def generate_markdown(self, results: list, market_status: str, holdings_status: list = None):
        """
        生成每日分析报告
        results: list of dict, 每个元素是一个标的的分析结果
        holdings_status: list of dict, 持仓监控结果
        """
        today = datetime.now().strftime("%Y-%m-%d")
        filename = self.report_dir / f"daily_report_{today}.md"
        
        # 按分数降序排列
        results.sort(key=lambda x: x['score'], reverse=True)
        
        content = []
        content.append(f"# 📅 量化交易日报 ({today})\n")
        content.append(f"### 🌊 市场状态: **{market_status}**\n")
        content.append("---\n")
        
        # 新增：持仓监控板块
        if holdings_status:
            content.append("## 🎒 持仓监控 (Holdings)\n")
            content.append("| 代码 | 名称 | 持仓天数 | 成本价 | 现价 | 浮盈 | 移动止盈线 | 状态 |\n")
            content.append("|---|---|---|---|---|---|---|---|\n")
            for h in holdings_status:
                pnl_str = f"{h['pnl_pct']:.2f}%"
                color = "🔴" if h['pnl_pct'] < 0 else "🟢"
                days = f"{h['days_held']}天" if h['days_held'] >= 0 else "-"
                row = f"| {h['code']} | {h['name']} | {days} | {h['buy_price']} | {h['current_price']} | {color} {pnl_str} | **{h['trailing_stop']}** | {h['status']} |"
                content.append(row + "\n")
            content.append("\n---\n")

        content.append("## 🚀 重点关注 (Score >= 0.6)\n")
        
        high_score_found = False
        for res in results:
            if res['score'] >= 0.6:
                high_score_found = True
                self._add_ticker_section(content, res)
        
        if not high_score_found:
            content.append("> ⚠️ 今日无高分标的，建议空仓或轻仓观望。\n")
            
        content.append("\n---\n")
        content.append("## 📋 所有标的概览\n")
        content.append("| 代码 | 名称 | 评分 | 建议 | 现价 | 止损位 | ATR |\n")
        content.append("|---|---|---|---|---|---|---|\n")
        
        for res in results:
            action = "🟢 买入" if res['is_buy'] else "⚪ 观望"
            stop_loss = res['risk']['initial_stop_loss'] if res['risk'] else "-"
            atr = res['risk']['atr'] if res['risk'] else "-"
            price = res['current_price']
            
            row = f"| {res['code']} | {res['name']} | **{res['score']}** | {action} | {price} | {stop_loss} | {atr} |"
            content.append(row + "\n")
            
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("".join(content))
            
        print(f"Report generated: {filename}")
        return filename

    def _add_ticker_section(self, content, res):
        risk = res['risk']
        reasons = res.get('reasons', [])
        
        content.append(f"### {res['name']} ({res['code']})\n")
        content.append(f"- **AI 评分**: {res['score']} / 1.0\n")
        content.append(f"- **当前价格**: {res['current_price']}\n")
        
        if reasons:
            content.append(f"- **📝 技术面点评**:\n")
            for reason in reasons:
                content.append(f"    - {reason}\n")
                
        if res['is_buy']:
            content.append(f"- **💡 决策建议**: **买入**\n")
            content.append(f"- **🛡️ 风控建议**:\n")
            content.append(f"    - 初始止损: **{risk['initial_stop_loss']}** (现价 - 2ATR)\n")
            content.append(f"    - 移动止盈: **{risk['trailing_stop_loss']}** (22日高点 - 2ATR)\n")
            content.append(f"    - 波动率(ATR): {risk['atr']}\n")
        else:
            content.append(f"- **💡 决策建议**: 观望 (评分未达标或大势不佳)\n")
        content.append("\n")

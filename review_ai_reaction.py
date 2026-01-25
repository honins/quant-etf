import pandas as pd
import os

def main():
    print("🕵️‍♂️ Reviewing AI Reaction Speed at Key Turning Points...\n")
    
    csv_path = "reports/daily_signals_last_60_days.csv"
    if not os.path.exists(csv_path):
        print("❌ Data file not found. Please run export_daily_signals.py first.")
        return
        
    df = pd.read_csv(csv_path)
    # 确保日期是字符串格式方便比较
    df['Date'] = df['Date'].astype(str)
    
    # 定义要复盘的案例
    cases = [
        {
            "name": "卫星ETF (159206.SZ)", 
            "code": "159206.SZ",
            "event": "🚀 主升浪启动",
            "key_date": "20251201", # 启动日
            "window": 5 # 前后5天
        },
        {
            "name": "新能源车ETF (515030.SH)",
            "code": "515030.SH",
            "event": "📈 底部反转",
            "key_date": "20251222", # 反转日
            "window": 5
        },
        {
            "name": "科创50ETF (588000.SH)",
            "code": "588000.SH",
            "event": "📉 顶部回调 (警示)",
            "key_date": "20260106", # 局部高点
            "window": 5
        }
    ]
    
    for case in cases:
        print(f"### 🎬 案例: {case['name']} - {case['event']}")
        print(f"🔑 关键日: {case['key_date']}\n")
        
        # 筛选数据
        target_df = df[df['Code'] == case['code']].sort_values('Date')
        
        # 找到关键日的索引
        try:
            key_idx = target_df[target_df['Date'] == case['key_date']].index[0]
            # 获取在原始df中的位置，以便iloc切片
            # 实际上直接用 date 过滤比较麻烦，不如转成 list 处理
            dates = target_df['Date'].tolist()
            if case['key_date'] not in dates:
                print(f"⚠️ Key date {case['key_date']} not found in data.")
                continue
                
            idx_in_list = dates.index(case['key_date'])
            start_idx = max(0, idx_in_list - case['window'])
            end_idx = min(len(dates), idx_in_list + case['window'] + 1)
            
            subset = target_df.iloc[start_idx:end_idx]
            
            print(f"| 日期 | 收盘价 | 涨跌幅 | **AI评分** | 信号 | 状态 |")
            print(f"|---|---|---|---|---|---|")
            
            for _, row in subset.iterrows():
                date = row['Date']
                price = row['Close']
                pct = row['PctChg']
                score = float(row['AI_Score'])
                signal = row['Signal']
                
                # 标记关键日
                mark = "👈 **启动/转折**" if date == case['key_date'] else ""
                
                # 评分趋势标记
                score_str = f"{score:.3f}"
                if score >= 0.6: score_str = f"**{score_str}** 🔥"
                elif score >= 0.45: score_str = f"{score_str} ✅"
                
                print(f"| {date} | {price} | {pct} | {score_str} | {signal} | {mark} |")
                
            print("\n" + "-"*60 + "\n")
            
        except IndexError:
            print("⚠️ Data error for this case.\n")

if __name__ == "__main__":
    main()

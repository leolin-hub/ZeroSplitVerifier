import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def mark_labels(data):
    
    LABEL_HOLD = '持有'
    LABEL_BUY = '買進'
    LABEL_SELL = '賣出'
    
    firstDay, lastDay = data.first_valid_index(), data.last_valid_index()
    peak = bottom = data['Close'][firstDay]  # 記錄前一波高低點
    buy = sell = False
    profit, numShares = 0, 0
    
    profits = np.empty(lastDay - firstDay + 1)
    profits[0] = 0
    data.at[firstDay, 'Label'] = LABEL_HOLD

    def mk_buying_label(i):
        nonlocal peak, buy
        label = False
        if data['Close'][i] >= peak:
            if buy is True:
                label = LABEL_BUY  # 選擇買進
                buy = False
            peak = data['Close'][i]  # 更新最高點
        else:
            if i > firstDay + 1:
                if data['Close'][i] < data['Close'][i-1] and data['Close'][i-2] < data['Close'][i-1]:  # 判斷昨日為波段高點
                    peak = data['Close'][i-1]  # 紀錄目前波段的高點
            buy = True
        return label

    def mk_selling_label(i):
        nonlocal bottom, sell
        label = False
        if data['Close'][i] <= bottom:
            if sell is True:
                label = LABEL_SELL  # 選擇賣出
                sell = False
            bottom = data['Close'][i]  # 更新最低點
        else:
            if i > firstDay + 1:
                if data['Close'][i] > data['Close'][i-1] and data['Close'][i-2] > data['Close'][i-1]:  # 判斷昨日為波段低點
                    bottom = data['Close'][i-1]  # 紀錄目前波段的低點
            sell = True
        return label

    # 迭代每個交易日
    for i in range(firstDay + 1, lastDay + 1):
        selling_label = mk_selling_label(i)
        buying_label = mk_buying_label(i)
        label = selling_label or buying_label or LABEL_HOLD

        if label == LABEL_SELL:  # 允許做空
            numShares -= 1
            profit = profit + data['Close'][i]
        elif label == LABEL_BUY:  # 允許融資
            numShares += 1
            profit = profit - data['Close'][i]

        profits[i - firstDay] = profit
        data.at[i, 'Label'] = label

    return data, profits

def show_labeling(stock, begin=None, end=None):
    
    if begin is None:
        begin = stock.first_valid_index()
    if end is None:
        end = stock.last_valid_index()
    
    # 選取數據區間
    if isinstance(begin, int) and isinstance(end, int):
        data = stock.iloc[begin:end+1].copy()
    else:
        data = stock[begin:end+1].copy()
    
    if data.empty:
        print("數據區間為空")
        return data
    
    # 執行標記
    data, profits = mark_labels(data)
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # === 上圖：股價和買賣點 ===
    # 繪製股價線
    ax1.plot(data.index, data['Close'], linewidth=2, color='#2E86C1', label='Close Price', alpha=0.8)
    
    # 標記買賣點
    buy_points = data[data['Label'] == '買進']
    sell_points = data[data['Label'] == '賣出']
    
    if not buy_points.empty:
        ax1.scatter(buy_points.index, buy_points['Close'], 
                   s=80, color='#28B463', marker='^', 
                   label=f'Buy ({len(buy_points)})', zorder=5, edgecolors='white', linewidth=1)
    
    if not sell_points.empty:
        ax1.scatter(sell_points.index, sell_points['Close'], 
                   s=80, color='#E74C3C', marker='v', 
                   label=f'Sell ({len(sell_points)})', zorder=5, edgecolors='white', linewidth=1)
    
    # 美化上圖
    ax1.set_ylabel('Stock Price', fontsize=12, fontweight='bold')
    ax1.set_title('Swing Trading', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#FAFAFA')
    
    # === 下圖：累積損益 ===
    # 繪製損益曲線
    profit_color = '#27AE60' if profits[-1] >= 0 else '#E74C3C'
    ax2.plot(range(len(profits)), profits, linewidth=3, color=profit_color, alpha=0.8)
    ax2.fill_between(range(len(profits)), profits, 0, alpha=0.2, color=profit_color)
    
    # 標記最高和最低點
    max_profit_idx = np.argmax(profits)
    min_profit_idx = np.argmin(profits)
    
    ax2.scatter(max_profit_idx, profits[max_profit_idx], 
               s=100, color='#28B463', marker='o', zorder=5, edgecolors='white', linewidth=2)
    ax2.scatter(min_profit_idx, profits[min_profit_idx], 
               s=100, color='#E74C3C', marker='o', zorder=5, edgecolors='white', linewidth=2)
    
    # 美化下圖
    ax2.set_ylabel('Accumulated Profit', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Days', fontsize=12, fontweight='bold')
    ax2.set_title('Accumulated Profits Curve', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#FAFAFA')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # 顯示累積的profits跟買賣次數
    if len(profits) > 0:
        final_profit = profits[-1] - profits[0]
        max_profit = np.max(profits)
        min_profit = np.min(profits)
        
        stats_text = f'Total Profits: {final_profit:.2f}\n' \
                    f'Max Profits: {max_profit:.2f}\n' \
                    f'Min Profits: {min_profit:.2f}\n' \
                    f'Total times: {len(buy_points) + len(sell_points)} '
        
        ax2.text(0.02, 0.98, stats_text, 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontsize=10, fontweight='bold')
    
    # 調整布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
    return data

def calculate_performance_metrics(data, profits):
    """
    計算交易績效指標
    """
    if len(profits) == 0:
        return {}
    
    buy_points = data[data['Label'] == '買進']
    sell_points = data[data['Label'] == '賣出']
    
    total_trades = len(buy_points) + len(sell_points)
    total_profit = profits[-1] - profits[0] if len(profits) > 1 else 0
    
    # 計算勝率
    trade_profits = []
    current_position = 0
    entry_price = 0
    
    for idx, row in data.iterrows():
        if row['Label'] == '買進':
            if current_position <= 0:
                entry_price = row['Close']
                current_position = 1
        elif row['Label'] == '賣出':
            if current_position > 0:
                trade_profit = row['Close'] - entry_price
                trade_profits.append(trade_profit)
                current_position = 0
    
    win_rate = 0
    avg_win = 0
    avg_loss = 0
    
    if trade_profits:
        winning_trades = [p for p in trade_profits if p > 0]
        losing_trades = [p for p in trade_profits if p < 0]
        
        win_rate = len(winning_trades) / len(trade_profits) * 100
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    metrics = {
        '總交易次數': total_trades,
        '買進次數': len(buy_points),
        '賣出次數': len(sell_points),
        '總損益': round(total_profit, 2),
        '勝率': f"{win_rate:.1f}%",
        '平均獲利': round(avg_win, 2),
        '平均虧損': round(avg_loss, 2),
        '最大獲利': round(np.max(profits), 2),
        '最大虧損': round(np.min(profits), 2)
    }
    
    return metrics

def load_and_process_data(filepath):
    """
    載入並處理CSV數據
    """
    try:
        df = pd.read_csv(filepath)
        
        if 'Close' not in df.columns:
            raise ValueError("CSV文件中缺少'Close'欄位")
        
        df.reset_index(drop=True, inplace=True)
        
        print(f"成功載入數據：{len(df)} 筆記錄")
        print(f"價格範圍：{df['Close'].min():.2f} ~ {df['Close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"載入數據時發生錯誤：{e}")
        return None

if __name__ == "__main__":
    # 載入數據
    csv_file = "C:/Users/zxczx/POPQORN/vanilla_rnn/utils/A1_bin.csv"  # 確保CSV文件在同一目錄下
    
    print("=== 波段交易策略分析 ===")
    
    # 載入股價數據
    stock_data = load_and_process_data(csv_file)
    
    if stock_data is not None:
        # 顯示收盤價統計分布
        print(f"\n數據概覽：")
        print(f"記錄數量: {len(stock_data)}")
        print(f"收盤價統計:")
        print(stock_data['Close'].describe())
        
        # 分析整個數據集
        print(f"\n正在分析整個數據集...")
        labeled_data = show_labeling(stock_data)
        
        # 計算績效指標
        _, profits = mark_labels(stock_data)
        metrics = calculate_performance_metrics(labeled_data, profits)
        
        print(f"\n=== 交易績效分析 ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # 也可以分析部分數據
        print(f"\n正在分析前1000筆數據...")
        sample_data = stock_data.head(12)
        sample_data = show_labeling(sample_data)
        sample_data.to_csv("labeled_data.csv", index=False)
        print("已將標記後的數據儲存為 labeled_data.csv")
        
    else:
        print("無法載入數據，請檢查CSV文件路徑和格式")
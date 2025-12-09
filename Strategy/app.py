import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="金银走势追踪（^.^）", layout="wide")

# --- 2. 核心功能：全盘搜索文件加载器 ---
def load_csv_data(code):
    """
    读取CSV并标准化列名，防止 'Close' 和 'close' 混用报错
    """
    target_filename = f"{code}.csv"
    found_path = None
    
    # 常用路径预设
    quick_paths = [
        f"Strategy/data/{target_filename}", 
        f"data/{target_filename}",
        f"{target_filename}"
    ]
    
    for path in quick_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    # 递归搜索
    if not found_path:
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            if target_filename in files:
                found_path = os.path.join(root, target_filename)
                break
    
    if found_path:
        try:
            df = pd.read_csv(found_path, index_col=0, parse_dates=True)
            # --- 数据清洗：统一列名为首字母大写 ---
            df.columns = [c.capitalize() for c in df.columns]
            return df, found_path
        except Exception as e:
            st.error(f"读取报错: {e}")
            return pd.DataFrame(), None
    else:
        return pd.DataFrame(), None

# --- 3. 策略逻辑引擎 ---
class StrategyEngine:
    def __init__(self, df):
        self.df = df.copy()

    def run_double_ma(self, short_w, long_w):
        """普通双均线策略"""
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        
        # 信号：快线 > 慢线 = 1 (多头), 否则 0 (空仓)
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff() # 1:买入, -1:卖出
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, short_w, long_w):
        """
        自动扶梯策略 (Escalator) - 基于之前的 Backtrader 逻辑复现
        """
        df = self.df.copy()
        
        # 1. 计算均线
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        
        # 2. 计算上轨(Max) 和 下轨(Min)
        df['kl_max'] = np.maximum(df['Line_Fast'], df['Line_Slow'])
        df['kl_min'] = np.minimum(df['Line_Fast'], df['Line_Slow'])

        # 3. 计算 K 线相对位置指标 (K%)
        # 公式: (Close - Low) / (High - Low)
        # 注意：Shift(2) 对应 Backtrader 的前第2根
        
        # --- Current (Shift 2) ---
        denom_cur = (df['High'].shift(2) - df['Low'].shift(2)).replace(0, np.nan) # 防除以0
        df['kl_range_cur'] = (df['Close'].shift(2) - df['Low'].shift(2)) / denom_cur
        
        # --- Previous (Shift 3) ---
        denom_pre = (df['High'].shift(3) - df['Low'].shift(3)).replace(0, np.nan) # 防除以0
        df['kl_range_pre'] = (df['Close'].shift(3) - df['Low'].shift(3)) / denom_pre

        # 4. 信号生成逻辑 (参考 Backtrader)
        # 逻辑：
        # - 买入：收盘价站上最大均线 且 K线形态符合特定要求（前值小，现值大->转强）
        # - 卖出：收盘价跌破最小均线 且 K线形态符合特定要求（前值大，现值小->转弱）
        
        cond_buy = (
            (df['Close'] > df['kl_max']) & 
            (df['kl_range_pre'] <= 0.25) & 
            (df['kl_range_cur'] > 0.75)
        )
        
        cond_sell = (
            (df['Close'] < df['kl_min']) & 
            (df['kl_range_pre'] >= 0.75) & 
            (df['kl_range_cur'] < 0.25)
        )

        # 使用 np.select 构建状态机
        # 1 = 持仓, 0 = 空仓/卖出, nan = 保持之前的状态
        conditions = [cond_buy, cond_sell]
        choices = [1, 0] 
        
        df['Raw_Signal'] = np.select(conditions, choices, default=np.nan)
        
        # 向下填充 (ffill) 保持持仓状态，直到遇到明确的卖出信号
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        
        # 计算买卖点 diff: 1为买入, -1为卖出
        df['Position'] = df['Signal'].diff()
        
        return df, df['kl_max'], df['kl_min']

# --- 4. 绘图函数 ---
def plot_chart(df, code, line1, line2, strategy_name):
    fig = go.Figure()
    
    # K线数据 (为了简化，这里只画收盘价线，也可以改为蜡烛图)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盘价', 
                             line=dict(color='gray', width=1, opacity=0.6)))
    
    # 策略线 (均线 或 通道)
    # 扶梯策略通常画阶梯状线，这里用 hv 形状模拟
    line_shape = 'hv' if "扶梯" in strategy_name else 'linear'
    
    fig.add_trace(go.Scatter(x=df.index, y=line1, name='上轨', 
                             line=dict(color='rgba(65, 105, 225, 0.8)', width=1.5, shape=line_shape)))
    fig.add_trace(go.Scatter(x=df.index, y=line2, name='下轨', 
                             line=dict(color='rgba(255, 140, 0, 0.8)', width=1.5, shape=line_shape)))
    
    # 填充通道颜色 (仅扶梯策略)
    if "扶梯" in strategy_name:
         fig.add_trace(go.Scatter(x=df.index, y=line1, fill=None, mode='lines', line_color='indigo', showlegend=False))
         fig.add_trace(go.Scatter(x=df.index, y=line2, fill='tonexty', mode='lines', line_color='indigo', 
                                  fillcolor='rgba(200, 200, 255, 0.1)', showlegend=False))

    # 买卖点标记
    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]

    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', 
                             marker=dict(symbol='triangle-up', size=12, color='red'), name='买入信号'))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', 
                             marker=dict(symbol='triangle-down', size=12, color='green'), name='卖出信号'))

    # 绘制盈亏连线 (仅当有成对交易时)
    # 简单的逻辑：每次买入找最近的一次卖出连线
    for bd, brow in buy.iterrows():
        subsequent_sells = sell[sell.index > bd]
        if not subsequent_sells.empty:
            sd = subsequent_sells.index[0]
            sp = subsequent_sells.loc[sd]['Close']
            bp = brow['Close']
            # 盈利红色，亏损绿色 (A股习惯)
            color = 'rgba(220,0,0,0.6)' if sp >= bp else 'rgba(0,128,0,0.6)'
            fig.add_trace(go.Scatter(x=[bd, sd], y=[bp, sp], mode='lines', 
                                     line=dict(color=color, width=2, dash='dot'), 
                                     showlegend=False, hoverinfo='skip'))

    fig.update_layout(
        title=f"{code} - {strategy_name}", 
        height=600, 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 5. 主程序 ---
def main():
    st.title("📈 ZC_金银趋势追踪")
    
    st.sidebar.header("⚙️ 策略配置")
    
    # 允许用户手动输入文件名，也可以选择预设
    target_code = st.sidebar.text_input("输入标的代码 (如 AU.SHF)", value="AU.SHF")
    
    strategy_type = st.sidebar.radio("选择策略模型", ["双均线策略 (Double MA)", "自动扶梯策略 (Escalator)"])

    # 加载数据
    df_raw, loaded_path = load_csv_data(target_code)

    if df_raw.empty:
        st.error(f"❌ 无法找到文件: {target_code}.csv")
        st.info("请确保CSV文件在当前目录或 'data' 文件夹下。")
        return
    else:
        st.success(f"已加载: {os.path.basename(loaded_path)} ({len(df_raw)} 条记录)")

    # 运行策略
    engine = StrategyEngine(df_raw)
    
    if "双均线" in strategy_type:
        st.sidebar.subheader("均线参数")
        short_w = st.sidebar.number_input("短周期 (Fast)", 5, 100, 10)
        long_w = st.sidebar.number_input("长周期 (Slow)", 20, 300, 50)
        df_res, l1, l2 = engine.run_double_ma(short_w, long_w)
    else:
        st.sidebar.subheader("扶梯参数")
        # 修正：这里需要两个参数对应 Backtrader 的 ma_slow 和 ma_fast
        fast_w = st.sidebar.number_input("均线1 (Fast)", 2, 100, 10)
        slow_w = st.sidebar.number_input("均线2 (Slow)", 10, 300, 50)
        df_res, l1, l2 = engine.run_escalator(fast_w, slow_w)
    
    # --- 结果展示区 ---
    last_row = df_res.iloc[-1]
    last_date = df_res.index[-1].strftime('%Y-%m-%d')
    
    # 状态判断
    current_pos = last_row['Signal']
    status_text = "持仓" if current_pos == 1 else "空仓 "
    status_color = "normal" if current_pos == 0 else "inverse"

    # 指标卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据日期", last_date)
    c2.metric("最新收盘", f"{last_row['Close']:.2f}")
    c3.metric("当前状态", status_text)
    
    # 计算最近一次操作后的收益
    last_signal_idx = df_res[df_res['Position'] != 0].index
    if not last_signal_idx.empty:
        last_op_date = last_signal_idx[-1]
        last_op_price = df_res.loc[last_op_date]['Close']
        last_op_type = "买入" if df_res.loc[last_op_date]['Position'] == 1 else "卖出"
        
        # 如果当前是持仓状态，计算浮动盈亏
        if current_pos == 1:
            pnl = (last_row['Close'] - last_op_price) / last_op_price * 100
            c4.metric(f"自 {last_op_date.strftime('%m-%d')} {last_op_type}", f"{pnl:.2f}%", delta=f"{pnl:.2f}%")
        else:
            c4.metric("最近操作", f"{last_op_date.strftime('%m-%d')} {last_op_type}")

    # 绘图
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type), use_container_width=True)
    
    # 详细数据表
    with st.expander("📊 查看详细信号记录"):
        # 筛选有动作的行
        signals = df_res[df_res['Position'] != 0].copy()
        if not signals.empty:
            signals['操作'] = signals['Position'].map({1: '🔺 买入', -1: '🔻 卖出'})
            cols_to_show = ['Close', '操作', 'Line_Fast', 'Line_Slow']
            if "扶梯" in strategy_type:
                cols_to_show = ['Close', '操作', 'kl_max', 'kl_min', 'kl_range_cur']
            
            st.dataframe(
                signals[cols_to_show].sort_index(ascending=False).style.format("{:.2f}"),
                use_container_width=True
            )
        else:
            st.write("当前区间内无交易信号")

if __name__ == "__main__":
    main()

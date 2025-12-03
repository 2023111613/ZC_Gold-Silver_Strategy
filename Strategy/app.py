# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="贵金属策略云端看板", layout="wide")

# --- 2. 核心功能：全盘搜索文件加载器 ---
def load_csv_data(code):
    """
    不猜路径了，直接在当前目录下递归搜索，只要文件存在就能找到
    """
    target_filename = f"{code}.csv"
    found_path = None
    
    # 1. 先尝试几个最可能的固定路径 (为了速度)
    quick_paths = [
        f"Strategy/data/{target_filename}",  # 你刚才提到的路径
        f"data/{target_filename}",
        f"{target_filename}"
    ]
    
    for path in quick_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    # 2. 如果固定路径没找到，启动“地毯式搜索” (os.walk)
    if not found_path:
        # os.getcwd() 获取当前工作目录，通常是仓库根目录
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            if target_filename in files:
                found_path = os.path.join(root, target_filename)
                break
    
    # 3. 读取结果
    if found_path:
        try:
            # print(f"Debug: Found file at {found_path}") # 调试用
            df = pd.read_csv(found_path, index_col=0, parse_dates=True)
            return df, found_path
        except Exception as e:
            st.error(f"找到了文件 ({found_path}) 但读取报错: {e}")
            return pd.DataFrame(), None
    else:
        return pd.DataFrame(), None

# --- 3. 策略逻辑引擎 ---
class StrategyEngine:
    def __init__(self, df):
        self.df = df.copy()

    def run_double_ma(self, short_w, long_w):
        """双均线策略"""
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff()
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, window):
        """电梯/突破策略"""
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=window).max().shift(1) 
        df['Line_Slow'] = df['Close'].rolling(window=window).min().shift(1) 
        
        conditions = [
            (df['Close'] > df['Line_Fast']),
            (df['Close'] < df['Line_Slow'])
        ]
        choices = [1, 0]
        df['Raw_Signal'] = np.select(conditions, choices, default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        return df, df['Line_Fast'], df['Line_Slow']

# --- 4. 绘图函数 ---
def plot_chart(df, code, line1, line2, strategy_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盘价', line=dict(color='gray', width=1)))
    
    line_shape = 'hv' if "电梯" in strategy_name else 'linear'
    fig.add_trace(go.Scatter(x=df.index, y=line1, name='快线/上轨', line=dict(color='blue', width=1, dash='dash', shape=line_shape)))
    fig.add_trace(go.Scatter(x=df.index, y=line2, name='慢线/下轨', line=dict(color='orange', width=1, dash='dash', shape=line_shape)))

    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]

    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='买入'))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='卖出'))

    for bd, brow in buy.iterrows():
        subsequent_sells = sell[sell.index > bd]
        if not subsequent_sells.empty:
            sd = subsequent_sells.index[0]
            sp = subsequent_sells.loc[sd]['Close']
            bp = brow['Close']
            color = 'rgba(220,0,0,0.8)' if sp >= bp else 'rgba(0,128,0,0.8)'
            fig.add_trace(go.Scatter(x=[bd, sd], y=[bp, sp], mode='lines', line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'))

    fig.update_layout(title=f"{code} - {strategy_name}", height=600, template="plotly_white", hovermode="x unified")
    return fig

# --- 5. 主程序 ---
def main():
    st.title("📈 贵金属量化策略 · 云端版")
    
    st.sidebar.header("⚙️ 策略配置")
    target_code = st.sidebar.selectbox("选择交易标的", ['AU.SHF', 'AG.SHF', 'Au9999.SGE'])
    strategy_type = st.sidebar.radio("选择策略模型", ["双均线策略 (Double MA)", "自动电梯策略 (Escalator)"])

    # 加载数据 (自动搜索路径)
    df_raw, loaded_path = load_csv_data(target_code)

    if df_raw.empty:
        st.error(f"❌ 无法找到文件: {target_code}.csv")
        st.warning("程序已尝试在所有子目录中搜索，但未找到。")
        st.info(f"当前搜索根目录: {os.getcwd()}")
        return

    # 运行策略
    engine = StrategyEngine(df_raw)
    
    if "双均线" in strategy_type:
        st.sidebar.subheader("均线参数")
        short_w = st.sidebar.number_input("短周期", 5, 60, 10)
        long_w = st.sidebar.number_input("长周期", 10, 200, 30)
        df_res, l1, l2 = engine.run_double_ma(short_w, long_w)
    else:
        st.sidebar.subheader("通道参数")
        window = st.sidebar.number_input("观察周期 (天)", 5, 100, 20)
        df_res, l1, l2 = engine.run_escalator(window)
    
    # 展示结果
    last_row = df_res.iloc[-1]
    last_date = df_res.index[-1].strftime('%Y-%m-%d')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("数据更新日期", last_date)
    col2.metric("最新收盘价", f"{last_row['Close']:.2f}")
    
    status = "持仓 (买入)" if last_row['Signal'] == 1 else "空仓 (卖出/观望)"
    col3.metric("当前建议", status, delta="多头" if last_row['Signal']==1 else "空仓")

    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type), use_container_width=True)
    
    with st.expander("查看详细信号记录"):
        signals = df_res[df_res['Position'] != 0].copy()
        if not signals.empty:
            signals['操作'] = signals['Position'].map({1: '买入', -1: '卖出'})
            st.dataframe(signals[['Close', '操作', 'Line_Fast', 'Line_Slow']].sort_index(ascending=False))

if __name__ == "__main__":
    main()

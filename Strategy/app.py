# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="贵金属策略云端看板", layout="wide")

# --- 2. 核心功能：超级文件加载器 (修复找不到文件的问题) ---
def load_csv_data(code):
    """
    尝试在多个层级寻找CSV文件，解决Streamlit Cloud路径与GitHub结构不一致的问题
    """
    # 定义所有可能的文件存放位置
    possible_paths = [
        f"{code}.csv",                # 1. 和代码在同一个文件夹
        f"../{code}.csv",             # 2. 在上一级文件夹 (GitHub根目录)
        f"data/{code}.csv",           # 3. 在 data 子文件夹
        f"../data/{code}.csv"         # 4. 在上一级的 data 子文件夹
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path:
        # 如果找到了，读取数据
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df, file_path
        except Exception as e:
            st.error(f"文件找到了 ({file_path}) 但读取出错: {e}")
            return pd.DataFrame(), None
    else:
        # 如果没找到，返回空
        return pd.DataFrame(), None

# --- 3. 策略逻辑引擎 (不依赖Wind，纯计算) ---
class StrategyEngine:
    def __init__(self, df):
        self.df = df.copy()

    def run_double_ma(self, short_w, long_w):
        """双均线策略"""
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        
        # 信号：快线 > 慢线
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff()
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, window):
        """电梯/突破策略"""
        df = self.df.copy()
        # 上轨：过去N天最高；下轨：过去N天最低 (shift(1)避免未来函数)
        df['Line_Fast'] = df['Close'].rolling(window=window).max().shift(1) 
        df['Line_Slow'] = df['Close'].rolling(window=window).min().shift(1) 
        
        conditions = [
            (df['Close'] > df['Line_Fast']), # 突破上轨买入
            (df['Close'] < df['Line_Slow'])  # 跌破下轨卖出
        ]
        choices = [1, 0]
        
        # 计算信号，使用ffill保持中间状态
        df['Raw_Signal'] = np.select(conditions, choices, default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        return df, df['Line_Fast'], df['Line_Slow']

# --- 4. 绘图函数 (带连线功能) ---
def plot_chart(df, code, line1, line2, strategy_name):
    fig = go.Figure()

    # 绘制基础K线和指标线
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盘价', line=dict(color='gray', width=1)))
    
    # 根据策略不同，线条样式微调 (电梯策略用阶梯线)
    line_shape = 'hv' if "电梯" in strategy_name else 'linear'
    fig.add_trace(go.Scatter(x=df.index, y=line1, name='快线/上轨', line=dict(color='blue', width=1, dash='dash', shape=line_shape)))
    fig.add_trace(go.Scatter(x=df.index, y=line2, name='慢线/下轨', line=dict(color='orange', width=1, dash='dash', shape=line_shape)))

    # 提取买卖点
    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]

    # 绘制买卖图标
    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='买入信号'))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='卖出信号'))

    # 绘制连线 (核心需求)
    for bd, brow in buy.iterrows():
        # 找到该买点后的第一个卖点
        subsequent_sells = sell[sell.index > bd]
        if not subsequent_sells.empty:
            sd = subsequent_sells.index[0]
            sp = subsequent_sells.loc[sd]['Close']
            bp = brow['Close']
            
            # 红色代表盈利，绿色代表亏损
            color = 'rgba(220,0,0,0.8)' if sp >= bp else 'rgba(0,128,0,0.8)'
            
            fig.add_trace(go.Scatter(
                x=[bd, sd], y=[bp, sp],
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title=f"{code} - {strategy_name} 回测图表", 
        height=600, 
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# --- 5. 主程序入口 ---
def main():
    st.title("📈 贵金属量化策略 · 云端版")
    
    # 侧边栏控制
    st.sidebar.header("⚙️ 策略配置")
    
    # 标的选择
    target_code = st.sidebar.selectbox("选择交易标的", ['AU.SHF', 'AG.SHF', 'Au9999.SGE'])
    
    # 策略选择
    strategy_type = st.sidebar.radio("选择策略模型", ["双均线策略 (Double MA)", "自动电梯策略 (Escalator)"])

    # 加载数据
    df_raw, loaded_path = load_csv_data(target_code)

    if df_raw.empty:
        st.error(f"❌ 未找到数据文件: {target_code}.csv")
        st.info("请确保你已经在本地运行了 '每日更新.bat'，且数据已成功上传到 GitHub。")
        st.warning(f"当前程序正在以下目录寻找文件: {os.getcwd()}")
        return

    # 显示数据来源（调试用，如果成功可以注释掉）
    # st.success(f"成功加载数据: {loaded_path}")

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
    
    # 顶部指标栏
    col1, col2, col3 = st.columns(3)
    col1.metric("数据日期", last_date)
    col2.metric("最新收盘价", f"{last_row['Close']:.2f}")
    
    status = "持仓 (买入)" if last_row['Signal'] == 1 else "空仓 (卖出/观望)"
    color = "normal" if last_row['Signal'] == 1 else "off"
    col3.metric("当前策略建议", status, delta="In Market" if last_row['Signal']==1 else "Out Market", delta_color=color)

    # 绘图
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type), use_container_width=True)
    
    # 底部数据表
    with st.expander("查看详细历史信号"):
        signals = df_res[df_res['Position'] != 0].copy()
        if not signals.empty:
            signals['操作'] = signals['Position'].map({1: '买入', -1: '卖出'})
            st.dataframe(signals[['Close', '操作', 'Line_Fast', 'Line_Slow']].sort_index(ascending=False))
        else:
            st.info("该时间段内无交易信号")

if __name__ == "__main__":
    main()

# 文件名: app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# 注意：云端不需要导入 WindPy，也不需要 try-except
# 我们直接读取 CSV 文件

st.set_page_config(page_title="贵金属策略看板", layout="wide")

# --- 策略类 (只保留计算逻辑，去掉Wind部分) ---
class Strategy_Engine:
    def __init__(self, df):
        self.df = df.copy()

    def double_ma(self, short_w, long_w):
        if self.df.empty: return self.df
        self.df['Line_Fast'] = self.df['Close'].rolling(short_w).mean()
        self.df['Line_Slow'] = self.df['Close'].rolling(long_w).mean()
        self.df['Signal'] = np.where(self.df['Line_Fast'] > self.df['Line_Slow'], 1, 0)
        self.df['Position'] = self.df['Signal'].diff()
        return self.df

    def escalator(self, window):
        if self.df.empty: return self.df
        self.df['Line_Fast'] = self.df['Close'].rolling(window).max().shift(1)
        self.df['Line_Slow'] = self.df['Close'].rolling(window).min().shift(1)
        
        conditions = [
            (self.df['Close'] > self.df['Line_Fast']),
            (self.df['Close'] < self.df['Line_Slow'])
        ]
        self.df['Raw_Signal'] = np.select(conditions, [1, 0], default=np.nan)
        self.df['Signal'] = self.df['Raw_Signal'].ffill().fillna(0)
        self.df['Position'] = self.df['Signal'].diff()
        return self.df

# --- 核心：读取数据的函数 ---
@st.cache_data
def load_data(code):
    """
    从 data 文件夹读取 CSV。
    在云端部署时，Streamlit 会直接读取仓库里的 data 文件夹。
    """
    file_path = f"data/{code}.csv"
    
    if not os.path.exists(file_path):
        st.error(f"未找到数据文件: {file_path}。请确认是否已同步到 GitHub。")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return df
    except Exception as e:
        st.error(f"读取数据出错: {e}")
        return pd.DataFrame()

# --- 绘图函数 (保持不变) ---
def plot_chart(df, code, strategy_name, line1_name, line2_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='收盘价', line=dict(color='gray', width=1)))
    
    line_shape = 'hv' if "Escalator" in strategy_name else 'linear'
    fig.add_trace(go.Scatter(x=df.index, y=df['Line_Fast'], mode='lines', name=line1_name, line=dict(color='blue', width=1, dash='dash', shape=line_shape)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Line_Slow'], mode='lines', name=line2_name, line=dict(color='orange', width=1, dash='dash', shape=line_shape)))

    buy_signals = df[df['Position'] == 1]
    sell_signals = df[df['Position'] == -1]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='买入', marker=dict(symbol='triangle-up', size=12, color='red')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='卖出', marker=dict(symbol='triangle-down', size=12, color='green')))

    # 画连线
    for buy_date, buy_row in buy_signals.iterrows():
        subsequent_sells = sell_signals[sell_signals.index > buy_date]
        if not subsequent_sells.empty:
            first_sell_date = subsequent_sells.index[0]
            first_sell_price = subsequent_sells.loc[first_sell_date]['Close']
            buy_price = buy_row['Close']
            line_color = 'rgba(214, 39, 40, 0.8)' if first_sell_price >= buy_price else 'rgba(44, 160, 44, 0.8)'
            fig.add_trace(go.Scatter(
                x=[buy_date, first_sell_date], y=[buy_price, first_sell_price],
                mode='lines', line=dict(color=line_color, width=4), showlegend=False, hoverinfo='skip'
            ))

    fig.update_layout(title=f'{code} - {strategy_name}', height=600, template="plotly_white", hovermode="x unified")
    return fig

# --- Main ---
def main():
    st.title("🌐 贵金属量化策略云端版")
    st.caption("数据来源: Wind (每日收盘后更新)")

    # 侧边栏配置
    st.sidebar.header("策略配置")
    target_code = st.sidebar.selectbox("交易标的", ['AU.SHF', 'AG.SHF', 'Au9999.SGE'])
    strategy_type = st.sidebar.radio("策略模型", ["双均线 (Double MA)", "自动电梯 (Escalator)"])

    # 加载数据
    df_raw = load_data(target_code)
    
    if df_raw.empty:
        return

    # 策略逻辑
    engine = Strategy_Engine(df_raw)
    
    if "Double MA" in strategy_type:
        short = st.sidebar.number_input("Short MA", 10)
        long_ma = st.sidebar.number_input("Long MA", 30)
        df = engine.double_ma(short, long_ma)
        l1, l2 = f"MA{short}", f"MA{long_ma}"
    else:
        window = st.sidebar.number_input("Window", 20)
        df = engine.escalator(window)
        l1, l2 = "上轨", "下轨"

    # 展示
    last_row = df.iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("最新日期", str(last_row.name.date()))
    col2.metric("最新价格", f"{last_row['Close']:.2f}")

    st.plotly_chart(plot_chart(df, target_code, strategy_type, l1, l2), use_container_width=True)
    
    with st.expander("查看数据源"):
        st.dataframe(df.tail(10).sort_index(ascending=False))

if __name__ == "__main__":
    main()

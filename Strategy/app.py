import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="金银走势追踪-专业版", layout="wide")

if 'opt_short' not in st.session_state: st.session_state.opt_short = None
if 'opt_long' not in st.session_state: st.session_state.opt_long = None

# --- 2. 数据加载与重采样 ---
def load_csv_data(code):
    target_filename = f"{code}.csv"
    found_path = None
    quick_paths = [f"Strategy/data/{target_filename}", f"data/{target_filename}", f"{target_filename}"]
    for path in quick_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if found_path:
        try:
            df = pd.read_csv(found_path, index_col=0, parse_dates=True)
            df.columns = df.columns.str.strip().str.lower()
            rename_map = {'close': 'Close', 'last': 'Close', 'price': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'vol': 'Volume'}
            df.rename(columns=rename_map, inplace=True)
            df.columns = [c.capitalize() for c in df.columns]
            return df.sort_index(), found_path
        except Exception as e:
            st.error(f"读取报错: {e}")
            return pd.DataFrame(), None
    return pd.DataFrame(), None

def resample_data(df, period_type):
    if df.empty or period_type == 'D': return df
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in df.columns: agg_dict['Volume'] = 'sum'
    return df.resample('W-FRI').agg(agg_dict).dropna()

# --- 3. 策略逻辑引擎 ---
class StrategyEngine:
    def __init__(self, df):
        self.df = df.copy()

    def run_double_ma(self, short_w, long_w):
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff() 
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, short_w, long_w):
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        df['kl_max'] = np.maximum(df['Line_Fast'], df['Line_Slow'])
        df['kl_min'] = np.minimum(df['Line_Fast'], df['Line_Slow'])
        
        # 扶梯逻辑
        denom = (df['High'].shift(1) - df['Low'].shift(1)).replace(0, np.nan)
        df['ratio_cur'] = (df['Close'].shift(1) - df['Low'].shift(1)) / denom
        df['ratio_pre'] = (df['Close'].shift(2) - df['Low'].shift(2)) / (df['High'].shift(2) - df['Low'].shift(2)).replace(0, np.nan)

        cond_buy = (df['Close'] > df['kl_max']) & (df['ratio_pre'] <= 0.25) & (df['ratio_cur'] > 0.75)
        cond_sell = (df['Close'] < df['kl_min']) & (df['ratio_pre'] >= 0.75) & (df['ratio_cur'] < 0.25)
        
        df['Raw_Signal'] = np.select([cond_buy, cond_sell], [1, 0], default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        return df, df['kl_max'], df['kl_min']

    def run_r_breaker(self, a=0.25, b=0.2, c=1.07,d=0.07):
        """
        R-Breaker 策略

        """
        df = self.df.copy()
        # 使用昨日(T-1)的价格计算今日的阈值
        high = df['High'].shift(1)
        low = df['Low'].shift(1)
        close = df['Close'].shift(1)
        
        # 计算 6 个价位
        df['Ssetup'] = high + a * (close - low)
        df['Bsetup'] = low - a * (high - close)
        df['Senter'] = b / 2 * (high + low) - c * low
        df['Benter'] = b / 2 * (high + low) - c * high
        df['Sbreak'] = df['Ssetup'] - d * (df['Ssetup'] - df['Bsetup'])
        df['Bbreak'] = df['Bsetup'] - d * (df['Ssetup'] - df['Bsetup'])
        
        # 简化的交易信号逻辑
        # 1. 趋势突破 (Trend)
        cond_buy_trend = df['Close'] > df['Sbreak']
        cond_sell_trend = df['Close'] < df['Bbreak']
        
        # 2. 反转逻辑 (Reversal)
        cond_sell_rev = (df['High'] > df['Ssetup']) & (df['Close'] < df['Senter'])
        cond_buy_rev = (df['Low'] < df['Bsetup']) & (df['Close'] > df['Benter'])
        
        df['Raw_Signal'] = np.nan
        df.loc[cond_buy_trend | cond_buy_rev, 'Raw_Signal'] = 1
        df.loc[cond_sell_trend | cond_sell_rev, 'Raw_Signal'] = 0
        
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        
        # 返回结果：为了绘图统一，返回两个最关键的突破边界
        return df, df['Sbreak'], df['Bbreak']

# --- 4. 优化与绘图 ---
def optimize_parameters(df, strategy_func, strat_name):
    """通用优化器"""
    best_count = -1
    best_params = (10, 50)
    
    if "R-Breaker" in strat_name:
        # R-Breaker 的优化逻辑可以针对 a 系数
        for a_val in np.linspace(0.2, 0.5, 10):
            res_df, _, _ = strategy_func(a=a_val)
            count = len(res_df[res_df['Position'] != 0])
            if count > best_count:
                best_count = count
                best_params = (a_val, 0.35) # 占位
        return best_params, best_count

    shorts = range(3, 35, 3)
    total_iters = len(shorts)
    my_bar = st.sidebar.progress(0, text="优化计算中...")
    for i, s_w in enumerate(shorts):
        for l_w in range(s_w + 5, 120, 10):
            res_df, _, _ = strategy_func(s_w, l_w)
            count = len(res_df[res_df['Position'] != 0])
            if count > best_count:
                best_count = count
                best_params = (s_w, l_w)
        my_bar.progress((i + 1) / total_iters)
    my_bar.empty()
    return best_params, best_count

def plot_chart(df, code, line1, line2, strategy_name, period_tag):
    fig = go.Figure()
    # K线基准
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盘价', line=dict(color='gray', width=1), opacity=0.5))
    
    # 策略线条
    if "R-Breaker" in strategy_name:
        fig.add_trace(go.Scatter(x=df.index, y=df['Sbreak'], name='突破卖出线(Sbreak)', line=dict(color='#FF1744', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Bbreak'], name='突破买入线(Bbreak)', line=dict(color='#00E676', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Ssetup'], name='观察阻力(Ssetup)', line=dict(color='#FFCDD2', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Bsetup'], name='观察支撑(Bsetup)', line=dict(color='#C8E6C9', width=1)))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=line1, name='指标线1', line=dict(color='#2962FF')))
        fig.add_trace(go.Scatter(x=df.index, y=line2, name='指标线2', line=dict(color='#FF6D00')))

    # 信号点
    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]
    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='red'), name='买入'))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='green'), name='卖出'))

    fig.update_layout(title=f"{code} - {strategy_name} ({period_tag})", height=650, template="plotly_white", hovermode="x unified")
    return fig

# --- 5. 主程序 ---
def main():
    st.title("📈 ZC_金银趋势 & 支撑阻力追踪")
    
    # 侧边栏
    ASSET_OPTIONS = {'AU.SHF': '黄金期货主力', 'AG.SHF': '白银期货主力', 'Au9999.SGE': '黄金现货9999'}
    target_code = st.sidebar.selectbox("交易标的", options=list(ASSET_OPTIONS.keys()), format_func=lambda x: ASSET_OPTIONS[x])
    
    period_mode = st.sidebar.radio("K线周期", ["日线 (Daily)", "周线 (Weekly)"], horizontal=True)
    is_weekly = "周线" in period_mode
    
    st.sidebar.markdown("---")
    strategy_type = st.sidebar.radio("模型类型", ["双均线策略 (Double MA)", "自动扶梯策略 (Escalator)", "R-Breaker 策略"])

    # 数据预处理
    df_raw, _ = load_csv_data(target_code)
    if df_raw.empty:
        st.error("无法加载数据，请检查文件是否存在")
        return
    df_active = resample_data(df_raw, 'W' if is_weekly else 'D')
    engine = StrategyEngine(df_active)

    # 策略参数区
    st.sidebar.subheader("⚙️ 参数设置")
    if "R-Breaker" in strategy_type:
        a_param = st.sidebar.slider("观察线系数 (a)", 0.1, 0.6, 0.35)
        b_param = st.sidebar.slider("反转线系数 (b)", 0.01, 0.2, 0.07)
        c_param = st.sidebar.slider("突破线系数 (c)", 0.1, 0.5, 0.25)
        df_res, l1, l2 = engine.run_r_breaker(a_param, b_param, c_param)
    else:
        # 优化按钮逻辑
        if st.sidebar.button("🔍 暴力搜索最佳周期"):
            strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
            best_p, _ = optimize_parameters(df_active, strat_func, strategy_type)
            st.session_state.opt_short, st.session_state.opt_long = best_p
        
        def_s = st.session_state.get('opt_short', 10) or 10
        def_l = st.session_state.get('opt_long', 50) or 50
        short_w = st.sidebar.number_input("快线周期", 2, 100, int(def_s))
        long_w = st.sidebar.number_input("慢线周期", 5, 200, int(def_l))
        
        strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
        df_res, l1, l2 = strat_func(short_w, long_w)

    # 显示看板
    last = df_res.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("最新价", f"{last['Close']:.2f}")
    c2.metric("当前状态", "多头持仓" if last['Signal'] == 1 else "空头/观望")
    c3.metric("信号总数", len(df_res[df_res['Position'] != 0]))

    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type, "周线" if is_weekly else "日线"), use_container_width=True)

    with st.expander("数据详情"):
        st.dataframe(df_res.tail(20).style.format("{:.2f}"))

if __name__ == "__main__":
    main()

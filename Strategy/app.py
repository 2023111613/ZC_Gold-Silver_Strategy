import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="金银走势追踪", layout="wide")

# 初始化所有可能的优化参数
for key in ['opt_short', 'opt_long', 'opt_a', 'opt_b', 'opt_c']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- 2. 数据加载与重采样 (保持不变) ---
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
            rename_map = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open'}
            df.rename(columns=rename_map, inplace=True)
            df.columns = [c.capitalize() for c in df.columns]
            return df.sort_index(), found_path
        except Exception as e:
            st.error(f"读取报错: {e}")
    return pd.DataFrame(), None

def resample_data(df, period_type):
    if df.empty or period_type == 'D': return df
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    return df.resample('W-FRI').agg(agg_dict).dropna()

# --- 3. 策略逻辑引擎 ---
class StrategyEngine:
    def __init__(self, df):
        self.df = df.copy()

    def run_double_ma(self, short_w, long_w):
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=int(short_w)).mean()
        df['Line_Slow'] = df['Close'].rolling(window=int(long_w)).mean()
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff() 
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, short_w, long_w):
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=int(short_w)).mean()
        df['Line_Slow'] = df['Close'].rolling(window=int(long_w)).mean()
        df['kl_max'] = np.maximum(df['Line_Fast'], df['Line_Slow'])
        df['kl_min'] = np.minimum(df['Line_Fast'], df['Line_Slow'])
        denom = (df['High'].shift(1) - df['Low'].shift(1)).replace(0, np.nan)
        df['r_cur'] = (df['Close'].shift(1) - df['Low'].shift(1)) / denom
        df['r_pre'] = (df['Close'].shift(2) - df['Low'].shift(2)) / (df['High'].shift(2) - df['Low'].shift(2)).replace(0, np.nan)
        cond_buy = (df['Close'] > df['kl_max']) & (df['r_pre'] <= 0.25) & (df['r_cur'] > 0.75)
        cond_sell = (df['Close'] < df['kl_min']) & (df['r_pre'] >= 0.75) & (df['r_cur'] < 0.25)
        df['Raw_Signal'] = np.select([cond_buy, cond_sell], [1, 0], default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        return df, df['kl_max'], df['kl_min']

    def run_r_breaker(self, a, b, c,d):
        df = self.df.copy()
        # 计算基于前一周期的数据
        h, l, c_prev = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        df['Ssetup'] = h + a * (c_prev - l)
        df['Bsetup'] = l - a * (h - c_prev)
        df['Senter'] = b / 2 * (h + l) - c * l
        df['Benter'] = b / 2 * (h + l) - c * h
        df['Sbreak'] = df['Ssetup'] - d * (df['Ssetup'] - df['Bsetup'])
        df['Bbreak'] = df['Bsetup'] + d* (df['Ssetup'] - df['Bsetup'])
        
        # 简化版逻辑：突破Sbreak买入，突破Bbreak卖出；或反转逻辑
        cond_buy = (df['Close'] > df['Sbreak']) | ((df['Low'] < df['Bsetup']) & (df['Close'] > df['Benter']))
        cond_sell = (df['Close'] < df['Bbreak']) | ((df['High'] > df['Ssetup']) & (df['Close'] < df['Senter']))
        
        df['Raw_Signal'] = np.select([cond_buy, cond_sell], [1, 0], default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        return df, df['Sbreak'], df['Bbreak']

# --- 4. 优化器 ---
def optimize_parameters(df, strategy_type, engine):
    best_count = -1
    best_params = {}
    
    progress_text = f"正在优化 {strategy_type} 参数..."
    my_bar = st.sidebar.progress(0, text=progress_text)

    if "R-Breaker" in strategy_type:
        # R-Breaker 搜索空间 (a, b, c)
        a_range = np.arange(0.2, 0.5, 0.05)
        b_range = np.arange(0.05, 0.15, 0.05)
        c_range = np.arange(0.15, 0.35, 0.05)
        d_range = np.arange(0.1,0.5,0.1)
        total = len(a_range)
        for i, a in enumerate(a_range):
            for b in b_range:
                for c in c_range:
                    for d in d_range:
                        res, _, _ = engine.run_r_breaker(a, b, c,d)
                        count = len(res[res['Position'] != 0])
                        if count > best_count:
                            best_count = count
                        best_params = {'a': a, 'b': b, 'c': c, 'd': d}
            my_bar.progress((i + 1) / total)
    else:
        # 均线类策略搜索空间
        shorts = range(5, 30, 3)
        total = len(shorts)
        for i, s_w in enumerate(shorts):
            for l_w in range(s_w + 10, 100, 10):
                strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
                res, _, _ = strat_func(s_w, l_w)
                count = len(res[res['Position'] != 0])
                if count > best_count:
                    best_count = count
                    best_params = {'short': s_w, 'long': l_w}
            my_bar.progress((i + 1) / total)
            
    my_bar.empty()
    return best_params, best_count

# --- 5. 绘图函数 (保持不变) ---
def plot_chart(df, code, line1, line2, strategy_name, period_tag):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='价格', line=dict(color='gray', width=1), opacity=0.4))
    
    if "R-Breaker" in strategy_name:
        fig.add_trace(go.Scatter(x=df.index, y=df['Sbreak'], name='突破卖出线', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Bbreak'], name='突破买入线', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Ssetup'], name='观察卖出价', line=dict(color='#FF6D00')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Bsetup'], name='观察买入价', line=dict(color='#2962FF')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Senter'], name='反转卖出价', line=dict(color="#8BEE12", dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['Benter'], name='反转买入价', line=dict(color="#FC08D3", dash='dash')))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=line1, name='快线/通道上沿', line=dict(color='#2962FF')))
        fig.add_trace(go.Scatter(x=df.index, y=line2, name='慢线/通道下沿', line=dict(color='#FF6D00')))

    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]
    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name='买入'))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'), name='卖出'))
    fig.update_layout(title=f"{code} {strategy_name}", height=600, template="plotly_white", hovermode="x unified")
    return fig

# --- 6. 主程序 ---
def main():
    st.title("📈ZC_金银走势追踪")
    
    # 侧边栏
    ASSET_OPTIONS = {'AU.SHF': '黄金期货', 'AG.SHF': '白银期货', 'Au9999.SGE': '黄金现货'}
    target_code = st.sidebar.selectbox("标的", options=list(ASSET_OPTIONS.keys()), format_func=lambda x: ASSET_OPTIONS[x])
    period_mode = st.sidebar.radio("周期", ["日线", "周线"], horizontal=True)
    strategy_type = st.sidebar.radio("策略", ["双均线策略 (Double MA)", "自动扶梯策略 (Escalator)", "R-Breaker 策略"])

    df_raw, _ = load_csv_data(target_code)
    df_active = resample_data(df_raw, 'W' if "周线" in period_mode else 'D')
    engine = StrategyEngine(df_active)

    # 优化按钮
    st.sidebar.markdown("---")
    if st.sidebar.button("🔍 搜索最优参数"):
        best_p, count = optimize_parameters(df_active, strategy_type, engine)
        if "R-Breaker" in strategy_type:
            st.session_state.opt_a, st.session_state.opt_b, st.session_state.opt_c = best_p['a'], best_p['b'], best_p['c']
        else:
            st.session_state.opt_short, st.session_state.opt_long = best_p['short'], best_p['long']
        st.toast(f"完成！找到信号数最多的组合", icon="✅")

    # 参数输入逻辑
    if "R-Breaker" in strategy_type:
        a = st.sidebar.slider("a (观察线)", 0.1, 0.6, st.session_state.opt_a or 0.35, 0.01)
        b = st.sidebar.slider("b (反转线)", 0.01, 0.3, st.session_state.opt_b or 0.07, 0.01)
        c = st.sidebar.slider("c (突破线)", 0.05, 0.5, st.session_state.opt_c or 0.25, 0.01)
        d = st.sidebar.slider("d (突破调整系数)", 0.1,1.0, st.session_state.opt_d or 0.3,0.05)
        df_res, l1, l2 = engine.run_r_breaker(a, b, c, d)
    else:
        s_w = st.sidebar.number_input("快线周期", 2, 100, int(st.session_state.opt_short or 10))
        l_w = st.sidebar.number_input("慢线周期", 5, 200, int(st.session_state.opt_long or 50))
        strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
        df_res, l1, l2 = strat_func(s_w, l_w)

    # 显示结果
    last = df_res.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("最新价", f"{last['Close']:.2f}")
    cols[1].metric("信号", "多头" if last['Signal']==1 else "观望")
    cols[2].metric("本周期信号数", len(df_res[df_res['Position'] != 0]))
    
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type, period_mode), use_container_width=True)

if __name__ == "__main__":
    main()

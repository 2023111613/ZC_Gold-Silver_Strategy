import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="ZC_金银走势追踪", layout="wide")

# 初始化所有可能的优化参数
for key in ['opt_short', 'opt_long', 'opt_a', 'opt_b', 'opt_c', 'opt_d', 'opt_k1', 'opt_k2', 'opt_n']:
    if key not in st.session_state:
        st.session_state[key] = None

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
            rename_map = {'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open'}
            df.rename(columns=rename_map, inplace=True)
            df.columns = [c.capitalize() for c in df.columns]
            return df.sort_index(), found_path
        except Exception as e:
            st.error(f"读取报错: {e}")
    return pd.DataFrame(), None

def resample_data(df, period_type):
    if df.empty or period_type == 'D': 
        return df
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    return df.resample('W-FRI').agg(agg_dict).dropna()

# --- 3. 回撤计算模块 ---
class DrawdownCalculator:
    """计算买卖信号之间的回撤"""
    
    @staticmethod
    def calculate_trade_drawdowns(df):
        """
        计算每笔交易的回撤信息
        返回: 包含回撤信息的DataFrame和汇总统计
        """
        if 'Position' not in df.columns or 'Signal' not in df.columns:
            return pd.DataFrame(), {}
        
        trades = []
        in_position = False
        entry_price = 0
        entry_date = None
        max_price_since_entry = 0
        min_price_since_entry = float('inf')
        max_drawdown = 0
        max_runup = 0
        
        for i, row in df.iterrows():
            if row['Position'] == 1:  # 买入信号
                in_position = True
                entry_price = row['Close']
                entry_date = i
                max_price_since_entry = row['Close']
                min_price_since_entry = row['Close']
                max_drawdown = 0
                max_runup = 0
                
            elif row['Position'] == -1 and in_position:  # 卖出信号
                exit_price = row['Close']
                exit_date = i
                pnl = (exit_price - entry_price) / entry_price * 100
                
                trades.append({
                    '入场日期': entry_date,
                    '入场价格': entry_price,
                    '出场日期': exit_date,
                    '出场价格': exit_price,
                    '持仓天数': (exit_date - entry_date).days,
                    '收益率(%)': round(pnl, 2),
                    '最大回撤(%)': round(max_drawdown, 2),
                    '最大浮盈(%)': round(max_runup, 2),
                    '盈亏比': round(abs(max_runup / max_drawdown), 2) if max_drawdown != 0 else float('inf')
                })
                in_position = False
                
            elif in_position:  # 持仓期间
                # 更新最高/最低价
                max_price_since_entry = max(max_price_since_entry, row['High'])
                min_price_since_entry = min(min_price_since_entry, row['Low'])
                
                # 计算从最高点回撤
                current_drawdown = (row['Low'] - max_price_since_entry) / max_price_since_entry * 100
                max_drawdown = min(max_drawdown, current_drawdown)
                
                # 计算从入场点的最大浮盈
                current_runup = (max_price_since_entry - entry_price) / entry_price * 100
                max_runup = max(max_runup, current_runup)
        
        # 处理仍在持仓的情况
        if in_position:
            last_row = df.iloc[-1]
            exit_price = last_row['Close']
            pnl = (exit_price - entry_price) / entry_price * 100
            trades.append({
                '入场日期': entry_date,
                '入场价格': entry_price,
                '出场日期': '持仓中',
                '出场价格': exit_price,
                '持仓天数': (df.index[-1] - entry_date).days,
                '收益率(%)': round(pnl, 2),
                '最大回撤(%)': round(max_drawdown, 2),
                '最大浮盈(%)': round(max_runup, 2),
                '盈亏比': round(abs(max_runup / max_drawdown), 2) if max_drawdown != 0 else float('inf')
            })
        
        trades_df = pd.DataFrame(trades)
        
        # 计算汇总统计
        stats = {}
        if len(trades_df) > 0:
            stats = {
                '总交易次数': len(trades_df),
                '盈利次数': len(trades_df[trades_df['收益率(%)'] > 0]),
                '亏损次数': len(trades_df[trades_df['收益率(%)'] < 0]),
                '胜率(%)': round(len(trades_df[trades_df['收益率(%)'] > 0]) / len(trades_df) * 100, 2),
                '平均收益率(%)': round(trades_df['收益率(%)'].mean(), 2),
                '平均回撤(%)': round(trades_df['最大回撤(%)'].mean(), 2),
                '最大单笔回撤(%)': round(trades_df['最大回撤(%)'].min(), 2),
                '平均持仓天数': round(trades_df[trades_df['持仓天数'].apply(lambda x: isinstance(x, (int, float)))]['持仓天数'].mean(), 1)
            }
        
        return trades_df, stats
    
    @staticmethod
    def calculate_equity_curve(df, initial_capital=100000):
        """计算权益曲线和动态回撤"""
        if 'Signal' not in df.columns:
            return df
        
        df = df.copy()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
        df['Equity'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
        
        # 计算动态回撤
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
        
        return df

# --- 4. 策略逻辑引擎 ---
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

    def run_r_breaker(self, a, b, c, d):
        df = self.df.copy()
        h, l, c_prev = df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1)
        
        df['Ssetup'] = h + a * (c_prev - l)
        df['Bsetup'] = l - a * (h - c_prev)
        df['Senter'] = b / 2 * (h + l) - c * l
        df['Benter'] = b / 2 * (h + l) - c * h
        df['Sbreak'] = df['Ssetup'] - d * (df['Ssetup'] - df['Bsetup'])
        df['Bbreak'] = df['Bsetup'] + d * (df['Ssetup'] - df['Bsetup'])
        
        cond_buy = (df['Close'] > df['Sbreak']) | ((df['Low'] < df['Bsetup']) & (df['Close'] > df['Benter']))
        cond_sell = (df['Close'] < df['Bbreak']) | ((df['High'] > df['Ssetup']) & (df['Close'] < df['Senter']))
        
        df['Raw_Signal'] = np.select([cond_buy, cond_sell], [1, 0], default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        return df, df['Sbreak'], df['Bbreak']

    def run_dual_thrust(self, n, k1, k2):

        df = self.df.copy()
        n = int(n)
        
        df['HH'] = df['High'].shift(1).rolling(window=n).max()  
        df['LL'] = df['Low'].shift(1).rolling(window=n).min()   
        df['HC'] = df['Close'].shift(1).rolling(window=n).max() 
        df['LC'] = df['Close'].shift(1).rolling(window=n).min() 
        
        df['Range'] = np.maximum(df['HH'] - df['LC'], df['HC'] - df['LL'])

        df['Upper_Band'] = df['Open'] + k1 * df['Range']
        df['Lower_Band'] = df['Open'] - k2 * df['Range']
        
        # 生成信号
        # 收盘价突破上轨做多，突破下轨平仓
        cond_buy = df['Close'] > df['Upper_Band']
        cond_sell = df['Close'] < df['Lower_Band']
        
        df['Raw_Signal'] = np.select([cond_buy, cond_sell], [1, 0], default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        
        return df, df['Upper_Band'], df['Lower_Band']

# --- 5. 优化器 ---
def optimize_parameters(df, strategy_type, engine):
    best_count = -1
    best_params = {}
    
    progress_text = f"正在优化 {strategy_type} 参数..."
    my_bar = st.sidebar.progress(0, text=progress_text)

    if "R-Breaker" in strategy_type:
        a_range = np.arange(0.2, 0.5, 0.1)
        b_range = np.arange(0.05, 0.15, 0.05)
        c_range = np.arange(0.15, 0.35, 0.1)
        d_range = np.arange(0.1, 0.5, 0.1)
        total = len(a_range)
        for i, a in enumerate(a_range):
            for b in b_range:
                for c in c_range:
                    for d in d_range:
                        res, _, _ = engine.run_r_breaker(a, b, c, d)
                        count = len(res[res['Position'] != 0])
                        if count > best_count:
                            best_count = count
                            best_params = {'a': a, 'b': b, 'c': c, 'd': d}
            my_bar.progress((i + 1) / total)
            
    elif "Dual Thrust" in strategy_type:
        n_range = range(3, 15, 2)
        k1_range = np.arange(0.3, 0.8, 0.1)
        k2_range = np.arange(0.3, 0.8, 0.1)
        total = len(n_range)
        for i, n in enumerate(n_range):
            for k1 in k1_range:
                for k2 in k2_range:
                    res, _, _ = engine.run_dual_thrust(n, k1, k2)
                    count = len(res[res['Position'] != 0])
                    if count > best_count:
                        best_count = count
                        best_params = {'n': n, 'k1': k1, 'k2': k2}
            my_bar.progress((i + 1) / total)
    else:
        shorts = range(5, 30, 5)
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

# --- 6. 绘图函数 ---
def plot_chart(df, code, line1, line2, strategy_name, period_tag, show_drawdown=False):
    if show_drawdown and 'Drawdown' in df.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.05, row_heights=[0.7, 0.3],
                           subplot_titles=(f"{code} {strategy_name} ({period_tag})", "动态回撤"))
    else:
        fig = go.Figure()
    
    # 基础K线价格
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='价格', 
                             line=dict(color='gray', width=1), opacity=0.4),
                  row=1 if show_drawdown and 'Drawdown' in df.columns else None,
                  col=1 if show_drawdown and 'Drawdown' in df.columns else None)
    
    if "R-Breaker" in strategy_name:
        fig.add_trace(go.Scatter(x=df.index, y=df['Sbreak'], name='突破卖出线(Sbreak)', 
                                line=dict(color='red', dash='dot')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bbreak'], name='突破买入线(Bbreak)', 
                                line=dict(color='green', dash='dot')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        fig.add_trace(go.Scatter(x=df.index, y=df['Ssetup'], name='观察卖出价', 
                                line=dict(color='#FF6D00')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        fig.add_trace(go.Scatter(x=df.index, y=df['Bsetup'], name='观察买入价', 
                                line=dict(color='#2962FF')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
    elif "Dual Thrust" in strategy_name:
        fig.add_trace(go.Scatter(x=df.index, y=line1, name='上轨(Upper Band)', 
                                line=dict(color='#E91E63', dash='dot')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        fig.add_trace(go.Scatter(x=df.index, y=line2, name='下轨(Lower Band)', 
                                line=dict(color='#4CAF50', dash='dot')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        # 添加Range区域填充
        fig.add_trace(go.Scatter(x=df.index, y=df['HH'], name='N日最高', 
                                line=dict(color='rgba(255,152,0,0.3)', width=1)),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        fig.add_trace(go.Scatter(x=df.index, y=df['LL'], name='N日最低', 
                                line=dict(color='rgba(33,150,243,0.3)', width=1)),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=line1, name='快线/通道上沿', 
                                line=dict(color='#2962FF')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
        fig.add_trace(go.Scatter(x=df.index, y=line2, name='慢线/通道下沿', 
                                line=dict(color='#FF6D00')),
                      row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)

    # 买卖信号点
    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]
    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', 
                            marker=dict(symbol='triangle-up', size=12, color='red'), name='买入信号'),
                  row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', 
                            marker=dict(symbol='triangle-down', size=12, color='green'), name='卖出信号'),
                  row=1 if show_drawdown and 'Drawdown' in df.columns else None, col=1 if show_drawdown and 'Drawdown' in df.columns else None)
    
    # 添加回撤图
    if show_drawdown and 'Drawdown' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name='回撤(%)', 
                                fill='tozeroy', line=dict(color='#F44336', width=1),
                                fillcolor='rgba(244,67,54,0.3)'),
                      row=2, col=1)
        fig.update_yaxes(title_text="回撤(%)", row=2, col=1)
    
    if show_drawdown and 'Drawdown' in df.columns:
        fig.update_layout(height=800, template="plotly_white", hovermode="x unified")
    else:
        fig.update_layout(title=f"{code} {strategy_name} ({period_tag})", 
                         height=600, template="plotly_white", hovermode="x unified")
    
    return fig

def plot_equity_curve(df):
    """绘制权益曲线"""
    if 'Equity' not in df.columns:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name='策略净值',
                            line=dict(color='#2196F3', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Peak'], name='历史峰值',
                            line=dict(color='#4CAF50', width=1, dash='dot')))
    
    fig.update_layout(title="策略权益曲线", height=400, template="plotly_white",
                     yaxis_title="净值", hovermode="x unified")
    return fig

# --- 7. 主程序 ---
def main():
    st.title("📈 ZC_金银走势追踪")
    
    # 侧边栏配置
    ASSET_OPTIONS = {'AU.SHF': '黄金期货', 'AG.SHF': '白银期货', 'Au9999.SGE': '黄金现货','TL.CFE': '30年国债主连'}
    target_code = st.sidebar.selectbox("选择标的", options=list(ASSET_OPTIONS.keys()), 
                                        format_func=lambda x: ASSET_OPTIONS[x])
    period_mode = st.sidebar.radio("选择周期", ["日线", "周线"], horizontal=True)
    strategy_type = st.sidebar.radio("选择策略", [
        "双均线策略 (Double MA)", 
        "自动扶梯策略 (Escalator)", 
        "R-Breaker 策略",
        "Dual Thrust 策略"
    ])
    
    # 新增：回撤分析开关
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 回撤分析")
    show_drawdown = st.sidebar.checkbox("显示动态回撤", value=True)
    show_trades = st.sidebar.checkbox("显示交易明细", value=True)
    show_equity = st.sidebar.checkbox("显示权益曲线", value=False)

    # 加载数据
    df_raw, path = load_csv_data(target_code)
    if df_raw.empty:
        st.error(f"未找到数据文件 {target_code}.csv")
        return

    df_active = resample_data(df_raw, 'W' if "周线" in period_mode else 'D')
    engine = StrategyEngine(df_active)

    # 优化按钮逻辑
    st.sidebar.markdown("---")
    if st.sidebar.button("🔍 搜索最优参数"):
        best_p, count = optimize_parameters(df_active, strategy_type, engine)
        if "R-Breaker" in strategy_type:
            st.session_state.opt_a = best_p['a']
            st.session_state.opt_b = best_p['b']
            st.session_state.opt_c = best_p['c']
            st.session_state.opt_d = best_p['d']
        elif "Dual Thrust" in strategy_type:
            st.session_state.opt_n = best_p['n']
            st.session_state.opt_k1 = best_p['k1']
            st.session_state.opt_k2 = best_p['k2']
        else:
            st.session_state.opt_short = best_p['short']
            st.session_state.opt_long = best_p['long']
        st.toast(f"参数优化完成！", icon="✅")

    # 参数动态调节区
    if "R-Breaker" in strategy_type:
        a = st.sidebar.slider("a (观察线宽度)", 0.1, 0.6, st.session_state.opt_a or 0.35, 0.01)
        b = st.sidebar.slider("b (反转线系数)", 0.01, 0.5, st.session_state.opt_b or 0.07, 0.01)
        c = st.sidebar.slider("c (反转偏置)", 0.05, 0.5, st.session_state.opt_c or 0.25, 0.01)
        d = st.sidebar.slider("d (突破调整系数)", 0.05, 1.0, st.session_state.opt_d or 0.30, 0.01)
        df_res, l1, l2 = engine.run_r_breaker(a, b, c, d)
    elif "Dual Thrust" in strategy_type:
        st.sidebar.markdown(" Dual Thrust 参数")
        n = st.sidebar.slider("N (回看周期)", 2, 20, int(st.session_state.opt_n or 5), 1,
                             help="计算Range的历史周期数")
        k1 = st.sidebar.slider("K1 (上轨系数)", 0.1, 1.0, float(st.session_state.opt_k1 or 0.5), 0.05,
                              help="上轨 = 开盘价 + K1 * Range")
        k2 = st.sidebar.slider("K2 (下轨系数)", 0.1, 1.0, float(st.session_state.opt_k2 or 0.5), 0.05,
                              help="下轨 = 开盘价 - K2 * Range")
        df_res, l1, l2 = engine.run_dual_thrust(n, k1, k2)
    else:
        s_w = st.sidebar.number_input("快线/短周期", 2, 100, int(st.session_state.opt_short or 10))
        l_w = st.sidebar.number_input("慢线/长周期", 5, 200, int(st.session_state.opt_long or 50))
        strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
        df_res, l1, l2 = strat_func(s_w, l_w)

    # 计算权益曲线和回撤
    df_res = DrawdownCalculator.calculate_equity_curve(df_res)
    trades_df, trade_stats = DrawdownCalculator.calculate_trade_drawdowns(df_res)

    # 显示指标面板
    last = df_res.iloc[-1]
    
    cols = st.columns(5)
    cols[0].metric("当前最新价", f"{last['Close']:.2f}")
    
    current_status = "多头持有" if last['Signal'] == 1 else "空仓观望"
    cols[1].metric("策略状态", current_status)
    
    signal_count = len(df_res[df_res['Position'] != 0])
    cols[2].metric("周期内交易次数", f"{signal_count} 次")
    
    # 新增：显示回撤相关指标
    if 'Drawdown' in df_res.columns:
        max_dd = df_res['Drawdown'].min()
        cols[3].metric("最大回撤", f"{max_dd:.2f}%")
    
    cols[4].metric("数据最后更新", last.name.strftime('%Y-%m-%d'))

    # 绘图
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type, period_mode, show_drawdown), 
                   use_container_width=True)
    
    # 显示权益曲线
    if show_equity:
        equity_fig = plot_equity_curve(df_res)
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)

    # 交易回撤分析面板
    if show_trades and len(trades_df) > 0:
        st.markdown("---")
        st.subheader("📉 交易回撤分析")
        
        # 统计信息卡片
        stat_cols = st.columns(4)
        stat_cols[0].metric("总交易次数", trade_stats.get('总交易次数', 0))
        stat_cols[1].metric("胜率", f"{trade_stats.get('胜率(%)', 0)}%")
        stat_cols[2].metric("平均收益率", f"{trade_stats.get('平均收益率(%)', 0)}%")
        stat_cols[3].metric("最大单笔回撤", f"{trade_stats.get('最大单笔回撤(%)', 0)}%")
        
        # 交易明细表
        st.markdown("📋 交易明细")
        
        # 格式化显示
        display_df = trades_df.copy()
        if len(display_df) > 0:
            # 添加颜色标记
            def highlight_pnl(val):
                if isinstance(val, (int, float)):
                    color = 'color: green' if val > 0 else 'color: red' if val < 0 else ''
                    return color
                return ''
            
            st.dataframe(
                display_df.style.applymap(highlight_pnl, subset=['收益率(%)']),
                use_container_width=True,
                height=300
            )
            
            # 下载按钮
            csv = trades_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下载交易记录",
                data=csv,
                file_name=f"{target_code}_{strategy_type}_trades.csv",
                mime="text/csv"
            )

    # 数据预览
    with st.expander("查看原始信号数据"):
        display_cols = ['Open', 'High', 'Low', 'Close', 'Signal', 'Position']
        if "Dual Thrust" in strategy_type:
            display_cols.extend(['Upper_Band', 'Lower_Band', 'Range', 'HH', 'LL'])
        if 'Drawdown' in df_res.columns:
            display_cols.append('Drawdown')
        
        available_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(df_res[available_cols].tail(20), use_container_width=True)

if __name__ == "__main__":
    main()



import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="金银走势追踪", layout="wide")

# 初始化 session_state 用于存储优化后的参数
if 'opt_short' not in st.session_state:
    st.session_state.opt_short = None
if 'opt_long' not in st.session_state:
    st.session_state.opt_long = None

# --- 2. 核心功能：数据加载与重采样 ---
def load_csv_data(code):
    """读取CSV并标准化列名"""
    target_filename = f"{code}.csv"
    found_path = None
    
    # 常用路径搜索
    quick_paths = [f"Strategy/data/{target_filename}", f"data/{target_filename}", f"{target_filename}"]
    for path in quick_paths:
        if os.path.exists(path):
            found_path = path
            break
            
    # 全盘搜索（如果没有找到）
    if not found_path:
        current_dir = os.getcwd()
        for root, dirs, files in os.walk(current_dir):
            if target_filename in files:
                found_path = os.path.join(root, target_filename)
                break
    
    if found_path:
        try:
            df = pd.read_csv(found_path, index_col=0, parse_dates=True)
            df.columns = df.columns.str.strip().str.lower()
            rename_map = {
                'close': 'Close', 'last': 'Close', 'price': 'Close',
                'high': 'High', 'max': 'High',
                'low': 'Low', 'min': 'Low',
                'open': 'Open',
                'vol': 'Volume', 'volume': 'Volume'
            }
            df.rename(columns=rename_map, inplace=True)
            df.columns = [c.capitalize() for c in df.columns]
            # 确保索引排序
            df = df.sort_index()
            return df, found_path
        except Exception as e:
            st.error(f"读取报错: {e}")
            return pd.DataFrame(), None
    else:
        return pd.DataFrame(), None

def resample_data(df, period_type):
    """
    将日线数据转换为周线数据
    period_type: 'D' (Daily) or 'W' (Weekly)
    """
    if df.empty or period_type == 'D':
        return df
    
    # 定义聚合逻辑
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'
        
    # 重采样并去除空值
    df_resampled = df.resample('W-FRI').agg(agg_dict).dropna()
    return df_resampled

# --- 3. 策略逻辑引擎 ---
class StrategyEngine:
    def __init__(self, df):
        self.df = df.copy()

    def run_double_ma(self, short_w, long_w):
        """普通双均线策略"""
        df = self.df.copy()
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff() 
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, short_w, long_w):
        """自动扶梯策略"""
        df = self.df.copy()
        
        # 简单校验
        if 'High' not in df.columns or 'Low' not in df.columns:
            return df, pd.Series(), pd.Series()
        
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        
        df['kl_max'] = np.maximum(df['Line_Fast'], df['Line_Slow'])
        df['kl_min'] = np.minimum(df['Line_Fast'], df['Line_Slow'])

        denom_cur = (df['High'].shift(1) - df['Low'].shift(1)).replace(0, np.nan)
        df['kl_range_cur'] = (df['Close'].shift(1) - df['Low'].shift(1)) / denom_cur
        
        denom_pre = (df['High'].shift(2) - df['Low'].shift(2)).replace(0, np.nan)
        df['kl_range_pre'] = (df['Close'].shift(2) - df['Low'].shift(2)) / denom_pre

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

        conditions = [cond_buy, cond_sell]
        choices = [1, 0] 
        
        df['Raw_Signal'] = np.select(conditions, choices, default=np.nan)
        df['Signal'] = df['Raw_Signal'].ffill().fillna(0)
        df['Position'] = df['Signal'].diff()
        
        return df, df['kl_max'], df['kl_min']

# --- 优化部分 ---
def optimize_parameters(df, strategy_func):
    """
    网格搜索：寻找产生交易信号次数最多的参数组合
    """
    best_count = -1
    best_params = (10, 50) # 默认
    
    # 定义搜索空间 (为了速度，步长设为稍大)
    # 快线: 3 ~ 30
    # 慢线: 10 ~ 100
    
    shorts = range(3, 35, 2)
    
    # 进度条
    progress_text = "正在后台暴力计算最佳波段参数..."
    my_bar = st.sidebar.progress(0, text=progress_text)
    total_iters = len(shorts)
    
    for i, s_w in enumerate(shorts):
        # 慢线必须大于快线
        longs = range(s_w + 5, 100, 5)
        for l_w in longs:
            # 运行策略
            res_df, _, _ = strategy_func(s_w, l_w)

            if 'Position' in res_df.columns:
                signal_count = len(res_df[res_df['Position'] != 0])
                
                # 更新最佳结果
                if signal_count > best_count:
                    best_count = signal_count
                    best_params = (s_w, l_w)
                    
        my_bar.progress((i + 1) / total_iters, text=progress_text)
        
    my_bar.empty()
    return best_params, best_count

# --- 5. 绘图函数 ---
def plot_chart(df, code, line1, line2, strategy_name, period_tag):
    fig = go.Figure()
    
    # 标题增加周期标识
    title_text = f"{code} [{period_tag}] - {strategy_name}"
    
    # 绘制K线 (如果需要更专业的图，可以用 go.Candlestick，这里为了保持风格统一用 Line + Fill)
    # 但由于有周线，建议如果数据量不大，画蜡烛图更清晰，这里保持原逻辑的基础做微调
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='收盘价', 
        opacity=0.6, line=dict(color='gray', width=1)
    ))
    
    if "扶梯" in strategy_name:
        # 扶梯通道绘制
        fig.add_trace(go.Scatter(
            x=df.index, y=line1, name='通道上沿', 
            fill=None, line=dict(color='#2962FF', width=1.5, shape='hv')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=line2, name='通道下沿', 
            fill='tonexty', fillcolor='rgba(83, 109, 254, 0.1)',
            line=dict(color='#00B0FF', width=1.5, shape='hv')
        ))
    else:
        # 双均线绘制
        fig.add_trace(go.Scatter(x=df.index, y=line1, name='快线', line=dict(color='#2962FF', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=line2, name='慢线', line=dict(color='#FF6D00', width=1.5)))

    # 买卖点
    if 'Position' in df.columns:
        buy = df[df['Position'] == 1]
        sell = df[df['Position'] == -1]
        
        fig.add_trace(go.Scatter(
            x=buy.index, y=buy['Close'], mode='markers', 
            marker=dict(symbol='triangle-up', size=12, color='#D50000', line=dict(width=1, color='white')), 
            name='买入'
        ))
        
        fig.add_trace(go.Scatter(
            x=sell.index, y=sell['Close'], mode='markers', 
            marker=dict(symbol='triangle-down', size=12, color='#00C853', line=dict(width=1, color='white')), 
            name='卖出'
        ))

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18)),
        height=600, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    )
    return fig

# --- 6. 主程序 ---
def main():
    st.title("📈ZC_金银趋势追踪")
    
    # --- 侧边栏配置 ---
    st.sidebar.header("全局设置")
    
    # 1. 标的选择
    ASSET_OPTIONS = {
        'AU.SHF': '黄金期货主力 (AU.SHF)',
        'AG.SHF': '白银期货主力 (AG.SHF)',
        'Au9999.SGE': '黄金现货9999 (Au9999.SGE)'
    }
    target_code = st.sidebar.selectbox(
        "交易标的", options=list(ASSET_OPTIONS.keys()),
        format_func=lambda x: ASSET_OPTIONS[x]
    )
    
    # 2. 周期切换 (日线/周线)
    st.sidebar.markdown("---")
    st.sidebar.subheader("周期选择")
    period_mode = st.sidebar.radio(
        "K线周期", 
        ["日线 (Daily)", "周线 (Weekly)"], 
        index=0,
        horizontal=True
    )
    is_weekly = "周线" in period_mode
    period_tag = "周线" if is_weekly else "日线"

    # 3. 策略选择
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 策略模型")
    strategy_type = st.sidebar.radio("模型类型", ["双均线策略 (Double MA)", "自动扶梯策略 (Escalator)"])

    # --- 数据处理 ---
    df_raw, _ = load_csv_data(target_code)
    
    if df_raw.empty:
        st.error(f"❌ 未找到 {target_code} 数据")
        return

    # 应用周期重采样
    df_active = resample_data(df_raw, 'W' if is_weekly else 'D')
    
    if len(df_active) < 50:
        st.warning(f"⚠️ 数据量不足 ({len(df_active)}行)，可能无法计算长期指标")

    engine = StrategyEngine(df_active)

    # --- 优化器与参数输入 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 参数调优")
    
    # 定义当前的策略函数引用，供优化器调用
    current_strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
    
    col_opt1, col_opt2 = st.sidebar.columns([3, 1])
    with col_opt1:
        st.info("💡点击计算交易频率最多参数")
    with col_opt2:
        if st.button("🔍"):
            best_p, best_c = optimize_parameters(df_active, current_strat_func)
            st.session_state.opt_short = best_p[0]
            st.session_state.opt_long = best_p[1]
            st.toast(f"优化完成！最佳参数: {best_p}, 信号数: {best_c}", icon="✅")

    # 确定默认值（优先使用 session_state 中的优化值）
    default_short = st.session_state.get('opt_short', 10) if st.session_state.get('opt_short') else 10
    default_long = st.session_state.get('opt_long', 50) if st.session_state.get('opt_long') else 50

    # 即使有优化值，用户也可以手动微调
    short_w = st.sidebar.number_input("快线周期 / Window 1", 2, 200, int(default_short))
    long_w = st.sidebar.number_input("慢线周期 / Window 2", 5, 500, int(default_long))
    
    # 如果用户手动修改了，清空优化状态以免下次混淆(可选)
    # st.session_state.opt_short = None 

    # --- 运行策略 ---
    df_res, l1, l2 = current_strat_func(short_w, long_w)

    # --- 结果展示区 ---
    last_row = df_res.iloc[-1]
    
    # 计算一些简单的统计
    total_signals = len(df_res[df_res['Position'] != 0])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前视图", period_tag)
    c2.metric("最新价格", f"{last_row['Close']:.2f}")
    
    pos_val = last_row['Signal']
    state_text = "多头持仓" if pos_val == 1 else "空头观望"
    c3.metric("当前信号", state_text)
    
    c4.metric("区间信号总数", f"{total_signals} 次", help="该参数组合下的买卖操作总次数")

    # 图表
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type, period_tag), use_container_width=True)
    
    # 详细数据
    with st.expander(f"📋 {period_tag} 详细交易记录"):
        signals = df_res[df_res['Position'] != 0].copy()
        if not signals.empty:
            signals['操作'] = signals['Position'].map({1: '🔺 买入', -1: '🔻 卖出'})
            cols = ['Close', '操作', 'Line_Fast', 'Line_Slow'] if 'Line_Fast' in signals.columns else ['Close', '操作']
            st.dataframe(
                signals[cols].sort_index(ascending=False).style.format("{:.2f}", subset=['Close'] + [c for c in cols if 'Line' in c]),
                use_container_width=True
            )
        else:
            st.write("当前参数下无交易信号")

if __name__ == "__main__":
    main()


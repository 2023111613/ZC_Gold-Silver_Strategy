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
    读取CSV并标准化列名
    """
    target_filename = f"{code}.csv"
    found_path = None
    
    quick_paths = [
        f"Strategy/data/{target_filename}", 
        f"data/{target_filename}",
        f"{target_filename}"
    ]
    
    for path in quick_paths:
        if os.path.exists(path):
            found_path = path
            break
            
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
                'close': 'Close', 'last': 'Close', 'price': 'Close', '收盘': 'Close', '收盘价': 'Close',
                'high': 'High', 'max': 'High', '最高': 'High', '最高价': 'High',
                'low': 'Low', 'min': 'Low', '最低': 'Low', '最低价': 'Low',
                'open': 'Open', '开盘': 'Open', '开盘价': 'Open',
                'vol': 'Volume', 'volume': 'Volume', '成交量': 'Volume'
            }
            df.rename(columns=rename_map, inplace=True)
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
        
        df['Signal'] = np.where(df['Line_Fast'] > df['Line_Slow'], 1, 0)
        df['Position'] = df['Signal'].diff() 
        return df, df['Line_Fast'], df['Line_Slow']

    def run_escalator(self, short_w, long_w):
        """自动扶梯策略"""
        df = self.df.copy()
        
        required_cols = ['High', 'Low']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"❌ 数据缺失：扶梯策略需要 {missing_cols} 列")
            st.stop()
        
        df['Line_Fast'] = df['Close'].rolling(window=short_w).mean()
        df['Line_Slow'] = df['Close'].rolling(window=long_w).mean()
        
        df['kl_max'] = np.maximum(df['Line_Fast'], df['Line_Slow'])
        df['kl_min'] = np.minimum(df['Line_Fast'], df['Line_Slow'])

        denom_cur = (df['High'].shift(2) - df['Low'].shift(2)).replace(0, np.nan)
        df['kl_range_cur'] = (df['Close'].shift(2) - df['Low'].shift(2)) / denom_cur
        
        denom_pre = (df['High'].shift(3) - df['Low'].shift(3)).replace(0, np.nan)
        df['kl_range_pre'] = (df['Close'].shift(3) - df['Low'].shift(3)) / denom_pre

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

# --- 4. 绘图函数 (视觉差异化升级版) ---
def plot_chart(df, code, line1, line2, strategy_name):
    fig = go.Figure()
    
    # 基础：绘制收盘价背景线
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['Close'], 
        name='收盘价', 
        opacity=0.5,
        line=dict(color='gray', width=1)
    ))
    
    # --- 🎨 核心差异化逻辑 ---
    
    if "扶梯" in strategy_name:
        # === 样式 B: 扶梯通道风格 ===
        # 特点：阶梯状连线 (shape='hv') + 中间填充 (Band)
        
        # 1. 绘制下轨 (Min) - 无填充，纯线
        fig.add_trace(go.Scatter(
            x=df.index, y=line2, 
            name='下轨 (Min)',
            line=dict(color='rgba(100, 100, 100, 0)', width=0), # 隐藏线条本身，只为了做填充边界
            showlegend=False
        ))
        
        # 2. 绘制上轨 (Max) - 填充到下轨
        fig.add_trace(go.Scatter(
            x=df.index, y=line1, 
            name='扶梯通道',
            fill='tonexty', # 填充到上一条线 (即下轨)
            fillcolor='rgba(83, 109, 254, 0.15)', # 淡淡的靛蓝色填充
            line=dict(color='rgba(83, 109, 254, 0.8)', width=1.5, shape='hv'), # 阶梯线
            mode='lines'
        ))
        
        # 为了让下轨也显示出线条（刚才隐藏了），再画一次下轨的线
        fig.add_trace(go.Scatter(
            x=df.index, y=line2, 
            name='下轨',
            line=dict(color='rgba(83, 109, 254, 0.8)', width=1.5, shape='hv'),
            showlegend=False
        ))
        
    else:
        # === 样式 A: 双均线交叉风格 ===
        # 特点：平滑曲线 + 鲜明的双色对比
        
        fig.add_trace(go.Scatter(
            x=df.index, y=line1, 
            name=' 快平均移动线', 
            line=dict(color='#2962FF', width=1.5) # 鲜艳蓝
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=line2, 
            name='慢平均移动线 ', 
            line=dict(color='#FF6D00', width=1.5) # 鲜艳橙
        ))

    # --- 通用：买卖点标记 ---
    buy = df[df['Position'] == 1]
    sell = df[df['Position'] == -1]

    # 买入图标 (红色向上三角)
    fig.add_trace(go.Scatter(
        x=buy.index, y=buy['Close'], mode='markers', 
        marker=dict(symbol='triangle-up', size=13, color='#D50000', line=dict(width=1, color='white')), 
        name='买入'
    ))
    
    # 卖出图标 (绿色向下三角)
    fig.add_trace(go.Scatter(
        x=sell.index, y=sell['Close'], mode='markers', 
        marker=dict(symbol='triangle-down', size=13, color='#00C853', line=dict(width=1, color='white')), 
        name='卖出'
    ))

    # --- 通用：盈亏连线 ---
    # 连接每一次买入和它之后最近的一次卖出
    for bd, brow in buy.iterrows():
        subsequent_sells = sell[sell.index > bd]
        if not subsequent_sells.empty:
            sd = subsequent_sells.index[0]
            sp = subsequent_sells.loc[sd]['Close']
            bp = brow['Close']
            
            # 盈利为红虚线，亏损为绿虚线
            line_color = 'rgba(213, 0, 0, 0.6)' if sp >= bp else 'rgba(0, 200, 83, 0.6)'
            
            fig.add_trace(go.Scatter(
                x=[bd, sd], y=[bp, sp], mode='lines', 
                line=dict(color=line_color, width=2, dash='dot'), 
                showlegend=False, hoverinfo='skip'
            ))

    fig.update_layout(
        title=dict(text=f"{code} - {strategy_name}", font=dict(size=20)),
        height=600, 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False), # 去掉X轴网格让图更清爽
        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    )
    return fig


# --- 5. 主程序 ---
def main():
    st.title("📈 ZC_金银趋势追踪")
    
    st.sidebar.header("⚙️ 策略配置")
    
    # 资产选择配置
    ASSET_OPTIONS = {
        'AU.SHF': '黄金期货主力 (AU.SHF)',
        'AG.SHF': '白银期货主力 (AG.SHF)',
        'Au9999.SGE': '黄金现货9999 (Au9999.SGE)'
    }
    
    target_code = st.sidebar.selectbox(
        "选择交易标的", 
        options=list(ASSET_OPTIONS.keys()),
        format_func=lambda x: ASSET_OPTIONS[x],
        index=0
    )
    
    strategy_type = st.sidebar.radio("选择策略模型", ["双均线策略 (Double MA)", "自动扶梯策略 (Escalator)"])

    # 加载数据
    df_raw, loaded_path = load_csv_data(target_code)

    if df_raw.empty:
        st.error(f"❌ 无法找到文件: {target_code}.csv")
        st.info("请运行 update_data.py 更新数据，或确保文件在 data 目录下。")
        return
    else:
        display_name = ASSET_OPTIONS.get(target_code, target_code)
        st.success(f"已加载: {display_name} ({len(df_raw)} 条记录)")

    # 运行策略
    engine = StrategyEngine(df_raw)
    
    if "双均线" in strategy_type:
        st.sidebar.subheader("均线参数")
        short_w = st.sidebar.number_input("快线周期", 5, 100, 10)
        long_w = st.sidebar.number_input("慢线周期", 20, 300, 50)
        df_res, l1, l2 = engine.run_double_ma(short_w, long_w)
    else:
        st.sidebar.subheader("通道参数")
        fast_w = st.sidebar.number_input("基准均线1", 2, 100, 10)
        slow_w = st.sidebar.number_input("基准均线2", 10, 300, 50)
        df_res, l1, l2 = engine.run_escalator(fast_w, slow_w)
    
    # --- 结果展示区 ---
    last_row = df_res.iloc[-1]
    last_date = df_res.index[-1].strftime('%Y-%m-%d')
    current_pos = last_row['Signal']
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据日期", last_date)
    c2.metric("最新收盘", f"{last_row['Close']:.2f}")
    c3.metric("当前状态", "持仓 (多头)" if current_pos == 1 else "空仓 (观望)", 
              delta="BULL" if current_pos==1 else "FLAT", delta_color="normal")
    
    # 收益计算
    last_signal_idx = df_res[df_res['Position'] != 0].index
    if not last_signal_idx.empty:
        last_op_date = last_signal_idx[-1]
        last_op_price = df_res.loc[last_op_date]['Close']
        last_op_type = "买入" if df_res.loc[last_op_date]['Position'] == 1 else "卖出"
        
        if current_pos == 1:
            pnl = (last_row['Close'] - last_op_price) / last_op_price * 100
            c4.metric(f"自 {last_op_date.strftime('%m-%d')} {last_op_type}", f"{pnl:.2f}%", delta=f"{pnl:.2f}%")
        else:
            c4.metric("最近操作", f"{last_op_date.strftime('%m-%d')} {last_op_type}")

    # 绘图
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type), use_container_width=True)
    
    # 信号表
    with st.expander("📊 查看详细信号记录"):
        signals = df_res[df_res['Position'] != 0].copy()
        if not signals.empty:
            signals['操作'] = signals['Position'].map({1: '🔺 买入', -1: '🔻 卖出'})
            
            if "扶梯" in strategy_type:
                cols_to_show = ['Close', '操作', 'kl_max', 'kl_min', 'kl_range_cur']
            else:
                cols_to_show = ['Close', '操作', 'Line_Fast', 'Line_Slow']
            
            df_display = signals[cols_to_show].sort_index(ascending=False)
            format_dict = {col: "{:.2f}" for col in cols_to_show if col != '操作'}
            
            st.dataframe(df_display.style.format(format_dict), use_container_width=True)
        else:
            st.write("当前区间内无交易信号")

if __name__ == "__main__":
    main()


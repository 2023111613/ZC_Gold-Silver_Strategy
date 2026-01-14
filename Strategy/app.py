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
@st.cache_data(ttl=3600)  # 缓存1小时后自动失效
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
                entry_price = row['High']
                entry_date = i
                max_price_since_entry = row['High']
                min_price_since_entry = row['Low']
                max_drawdown = 0
                max_runup = 0
                
            elif row['Position'] == -1 and in_position:  # 卖出信号
                exit_price = row['High']
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
            exit_price = last_row['High']
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
        df['Daily_Return'] = df['High'].pct_change()
        df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
        df['Equity'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
        
        # 计算动态回撤
        df['Peak'] = df['Equity'].cummax()
        df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak'] * 100
        
        return df

# --- 4. 回测引擎 ---
class BacktestEngine:
    """
    完整的回测引擎，支持:
    - 初始资金设置
    - 手续费和滑点
    - 仓位管理
    - 详细的交易记录
    - 多维度业绩分析
    """
    
    def __init__(self, initial_capital=100000, commission_rate=0.0003, slippage=0.0001):
        """
        初始化回测引擎
        参数:
            initial_capital: 初始资金
            commission_rate: 手续费率 (默认万3)
            slippage: 滑点 (默认万1)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
    def run_backtest(self, df, position_size=1.0):
        """
        运行回测
        参数:
            df: 包含Signal和Position列的DataFrame
            position_size: 仓位比例 (0-1)
        返回:
            backtest_result: 包含所有回测结果的字典
        """
        if 'Signal' not in df.columns or 'Position' not in df.columns:
            return None
        
        df = df.copy()
        
        # 初始化变量
        capital = self.initial_capital
        position = 0  # 当前持仓数量
        entry_price = 0
        trades = []
        equity_curve = []
        daily_returns = []
        
        # 逐日回测
        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row['Close']
            
            # 记录每日权益
            if position > 0:
                current_value = capital + position * current_price
            else:
                current_value = capital
            equity_curve.append({
                'date': date,
                'equity': current_value,
                'capital': capital,
                'position_value': position * current_price if position > 0 else 0
            })
            
            # 计算日收益率
            if i > 0:
                prev_equity = equity_curve[-2]['equity']
                daily_ret = (current_value - prev_equity) / prev_equity if prev_equity > 0 else 0
                daily_returns.append(daily_ret)
            
            # 处理交易信号
            if row['Position'] == 1 and position == 0:  # 买入信号
                # 计算可买入数量
                buy_price = current_price * (1 + self.slippage)
                commission = capital * position_size * self.commission_rate
                available_capital = capital * position_size - commission
                position = available_capital / buy_price
                entry_price = buy_price
                capital = capital * (1 - position_size)
                
                trades.append({
                    'type': 'BUY',
                    'date': date,
                    'price': buy_price,
                    'quantity': position,
                    'value': position * buy_price,
                    'commission': commission,
                    'capital_after': capital
                })
                
            elif row['Position'] == -1 and position > 0:  # 卖出信号
                sell_price = current_price * (1 - self.slippage)
                sell_value = position * sell_price
                commission = sell_value * self.commission_rate
                
                # 计算本次交易盈亏
                pnl = (sell_price - entry_price) * position - commission
                pnl_pct = (sell_price - entry_price) / entry_price * 100
                
                capital += sell_value - commission
                
                trades.append({
                    'type': 'SELL',
                    'date': date,
                    'price': sell_price,
                    'quantity': position,
                    'value': sell_value,
                    'commission': commission,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': capital
                })
                
                position = 0
                entry_price = 0
        
        # 如果最后还有持仓，按最后价格结算
        if position > 0:
            last_price = df['Close'].iloc[-1] * (1 - self.slippage)
            sell_value = position * last_price
            commission = sell_value * self.commission_rate
            pnl = (last_price - entry_price) * position - commission
            pnl_pct = (last_price - entry_price) / entry_price * 100
            capital += sell_value - commission
            
            trades.append({
                'type': 'SELL (结算)',
                'date': df.index[-1],
                'price': last_price,
                'quantity': position,
                'value': sell_value,
                'commission': commission,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'capital_after': capital
            })
        
        # 创建权益曲线DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # 计算业绩指标
        metrics = self._calculate_metrics(equity_df, daily_returns, trades)
        
        # 创建交易记录DataFrame
        trades_df = self._create_trades_df(trades)
        
        return {
            'equity_df': equity_df,
            'trades_df': trades_df,
            'metrics': metrics,
            'final_capital': capital,
            'daily_returns': daily_returns
        }
    
    def _calculate_metrics(self, equity_df, daily_returns, trades):
        """计算回测业绩指标"""
        if len(equity_df) == 0:
            return {}
        
        # 基础指标
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100
        
        # 计算最大回撤
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # 找到最大回撤期间
        max_dd_end_idx = equity_df['drawdown'].idxmin()
        max_dd_start_idx = equity_df.loc[:max_dd_end_idx, 'equity'].idxmax()
        
        # 计算夏普率
        if len(daily_returns) > 0:
            returns_arr = np.array(daily_returns)
            annual_return = np.mean(returns_arr) * 252
            annual_vol = np.std(returns_arr) * np.sqrt(252)
            sharpe = (annual_return - 0.03) / annual_vol if annual_vol > 0 else 0
        else:
            sharpe = 0
            annual_return = 0
            annual_vol = 0
        
        # 交易统计
        sell_trades = [t for t in trades if t['type'].startswith('SELL')]
        if len(sell_trades) > 0:
            win_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
            lose_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
            win_rate = len(win_trades) / len(sell_trades) * 100
            
            avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
            avg_lose = np.mean([t['pnl'] for t in lose_trades]) if lose_trades else 0
            profit_factor = abs(sum([t['pnl'] for t in win_trades]) / sum([t['pnl'] for t in lose_trades])) if lose_trades and sum([t['pnl'] for t in lose_trades]) != 0 else float('inf')
            
            max_win = max([t.get('pnl_pct', 0) for t in sell_trades])
            max_lose = min([t.get('pnl_pct', 0) for t in sell_trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_lose = 0
            profit_factor = 0
            max_win = 0
            max_lose = 0
        
        # 计算总手续费
        total_commission = sum([t.get('commission', 0) for t in trades])
        
        # 计算Calmar比率
        calmar = abs(annual_return / max_drawdown * 100) if max_drawdown != 0 else 0
        
        # 计算年化天数
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365 if days > 0 else 1
        annual_return_pct = ((final_equity / initial_equity) ** (1/years) - 1) * 100 if years > 0 else total_return
        
        return {
            '初始资金': initial_equity,
            '最终资金': final_equity,
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annual_return_pct, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '最大回撤开始': max_dd_start_idx.strftime('%Y-%m-%d') if hasattr(max_dd_start_idx, 'strftime') else str(max_dd_start_idx),
            '最大回撤结束': max_dd_end_idx.strftime('%Y-%m-%d') if hasattr(max_dd_end_idx, 'strftime') else str(max_dd_end_idx),
            '夏普率': round(sharpe, 2),
            'Calmar比率': round(calmar, 2),
            '年化波动率(%)': round(annual_vol * 100, 2),
            '交易次数': len(sell_trades),
            '胜率(%)': round(win_rate, 2),
            '平均盈利': round(avg_win, 2),
            '平均亏损': round(avg_lose, 2),
            '盈亏比': round(abs(avg_win/avg_lose), 2) if avg_lose != 0 else float('inf'),
            '盈利因子': round(profit_factor, 2),
            '最大单笔盈利(%)': round(max_win, 2),
            '最大单笔亏损(%)': round(max_lose, 2),
            '总手续费': round(total_commission, 2),
            '回测天数': days
        }
    
    def _create_trades_df(self, trades):
        """创建交易记录DataFrame"""
        if not trades:
            return pd.DataFrame()
        
        records = []
        buy_trade = None
        
        for trade in trades:
            if trade['type'] == 'BUY':
                buy_trade = trade
            elif trade['type'].startswith('SELL') and buy_trade:
                records.append({
                    '入场日期': buy_trade['date'],
                    '入场价格': round(buy_trade['price'], 2),
                    '出场日期': trade['date'],
                    '出场价格': round(trade['price'], 2),
                    '数量': round(buy_trade['quantity'], 4),
                    '盈亏金额': round(trade.get('pnl', 0), 2),
                    '收益率(%)': round(trade.get('pnl_pct', 0), 2),
                    '手续费': round(buy_trade['commission'] + trade['commission'], 2)
                })
                buy_trade = None
        
        return pd.DataFrame(records)


def plot_backtest_result(backtest_result, benchmark_df=None):
    """绘制回测结果图表"""
    equity_df = backtest_result['equity_df']
    
    # 创建子图
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                       subplot_titles=('策略净值曲线', '回撤', '持仓价值'))
    
    # 第一行：净值曲线
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['equity'], 
                            name='策略净值', line=dict(color='#2196F3', width=2)),
                  row=1, col=1)
    
    # 添加基准对比（如果有）
    if benchmark_df is not None and 'Close' in benchmark_df.columns:
        initial_price = benchmark_df['Close'].iloc[0]
        initial_equity = equity_df['equity'].iloc[0]
        benchmark_equity = benchmark_df['Close'] / initial_price * initial_equity
        fig.add_trace(go.Scatter(x=benchmark_df.index, y=benchmark_equity,
                                name='买入持有', line=dict(color='#9E9E9E', width=1, dash='dot')),
                      row=1, col=1)
    
    # 峰值线
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['peak'],
                            name='历史峰值', line=dict(color='#4CAF50', width=1, dash='dash')),
                  row=1, col=1)
    
    # 第二行：回撤
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['drawdown'],
                            name='回撤(%)', fill='tozeroy',
                            line=dict(color='#F44336', width=1),
                            fillcolor='rgba(244,67,54,0.3)'),
                  row=2, col=1)
    
    # 第三行：持仓价值
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df['position_value'],
                            name='持仓价值', fill='tozeroy',
                            line=dict(color='#FF9800', width=1),
                            fillcolor='rgba(255,152,0,0.3)'),
                  row=3, col=1)
    
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤(%)", row=2, col=1)
    fig.update_yaxes(title_text="持仓价值", row=3, col=1)
    
    fig.update_layout(height=700, template="plotly_white", hovermode="x unified",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig


def plot_monthly_returns(backtest_result):
    """绘制月度收益热力图"""
    equity_df = backtest_result['equity_df']
    
    # 计算月度收益
    monthly_equity = equity_df['equity'].resample('ME').last()
    monthly_returns = monthly_equity.pct_change() * 100
    
    # 创建年月矩阵
    returns_matrix = []
    years = monthly_returns.index.year.unique()
    months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    
    for year in years:
        year_returns = []
        for month in range(1, 13):
            try:
                ret = monthly_returns[(monthly_returns.index.year == year) & 
                                      (monthly_returns.index.month == month)]
                year_returns.append(ret.values[0] if len(ret) > 0 else None)
            except:
                year_returns.append(None)
        returns_matrix.append(year_returns)
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=returns_matrix,
        x=months,
        y=[str(y) for y in years],
        colorscale='RdYlGn',
        zmid=0,
        text=[[f'{v:.1f}%' if v is not None else '' for v in row] for row in returns_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y}年%{x}: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='月度收益热力图',
        height=max(300, len(years) * 40 + 100),
        template="plotly_white"
    )
    
    return fig


def plot_trade_distribution(trades_df):
    """绘制交易收益分布图"""
    if trades_df.empty or '收益率(%)' not in trades_df.columns:
        return None
    
    returns = trades_df['收益率(%)']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('收益率分布', '累计盈亏'))
    
    # 收益率直方图
    fig.add_trace(go.Histogram(x=returns, nbinsx=20, name='收益分布',
                               marker_color='#2196F3'), row=1, col=1)
    
    # 添加零线
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 累计盈亏
    cumulative_pnl = trades_df['盈亏金额'].cumsum()
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_pnl)+1)), y=cumulative_pnl,
                            mode='lines+markers', name='累计盈亏',
                            line=dict(color='#4CAF50', width=2)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    fig.update_xaxes(title_text="收益率(%)", row=1, col=1)
    fig.update_xaxes(title_text="交易序号", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=1, col=1)
    fig.update_yaxes(title_text="累计盈亏", row=1, col=2)
    
    fig.update_layout(height=400, template="plotly_white", showlegend=False)
    
    return fig


# --- 5. 策略逻辑引擎 ---
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
def calculate_sharpe_ratio(df, risk_free_rate=0.03):
    """
    计算策略的夏普率
    参数:
        df: 包含Signal列的DataFrame
        risk_free_rate: 年化无风险利率，默认3%
    返回:
        夏普率（年化）
    """
    if 'Signal' not in df.columns or len(df) < 2:
        return -np.inf
    
    df = df.copy()
    df['Daily_Return'] = df['High'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
    
    # 去除NaN
    strategy_returns = df['Strategy_Return'].dropna()
    
    if len(strategy_returns) < 2 or strategy_returns.std() == 0:
        return -np.inf
    
    # 年化参数（假设252个交易日）
    trading_days = 252
    
    # 计算年化收益率和年化波动率
    mean_return = strategy_returns.mean() * trading_days
    std_return = strategy_returns.std() * np.sqrt(trading_days)
    
    # 计算夏普率
    sharpe = (mean_return - risk_free_rate) / std_return if std_return > 0 else -np.inf
    
    return sharpe

def calculate_total_return(df):
    """计算策略总收益率"""
    if 'Signal' not in df.columns or len(df) < 2:
        return -np.inf
    
    df = df.copy()
    df['Daily_Return'] = df['High'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
    
    # 计算累计收益
    total_return = (1 + df['Strategy_Return'].fillna(0)).prod() - 1
    return total_return * 100  # 转为百分比

def calculate_max_drawdown(df):
    """计算最大回撤（返回负值，越小越好，优化时取负）"""
    if 'Signal' not in df.columns or len(df) < 2:
        return -np.inf
    
    df = df.copy()
    df['Daily_Return'] = df['High'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
    df['Equity'] = (1 + df['Strategy_Return'].fillna(0)).cumprod()
    df['Peak'] = df['Equity'].cummax()
    df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
    
    max_dd = df['Drawdown'].min()
    # 返回负的最大回撤，这样优化时越大越好（回撤越小越好）
    return -max_dd * 100  # 返回正值百分比

def evaluate_strategy(df, metric='sharpe'):
    """根据指定指标评估策略"""
    if metric == 'sharpe':
        return calculate_sharpe_ratio(df)
    elif metric == 'return':
        return calculate_total_return(df)
    elif metric == 'drawdown':
        return calculate_max_drawdown(df)
    elif metric == 'trade_count':
        return len(df[df['Position'] != 0])
    else:
        return calculate_sharpe_ratio(df)

def optimize_parameters(df, strategy_type, engine, optimize_metric='sharpe'):
    best_score = -np.inf
    best_params = {}
    
    metric_names = {
        'sharpe': '夏普率',
        'return': '总收益率',
        'drawdown': '最小回撤',
        'trade_count': '交易次数'
    }
    
    progress_text = f"正在优化 {strategy_type} 参数 (目标: {metric_names.get(optimize_metric, optimize_metric)})..."
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
                        score = evaluate_strategy(res, optimize_metric)
                        if score > best_score:
                            best_score = score
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
                    score = evaluate_strategy(res, optimize_metric)
                    if score > best_score:
                        best_score = score
                        best_params = {'n': n, 'k1': k1, 'k2': k2}
            my_bar.progress((i + 1) / total)
    else:
        shorts = range(5, 30, 5)
        total = len(shorts)
        for i, s_w in enumerate(shorts):
            for l_w in range(s_w + 10, 100, 10):
                strat_func = engine.run_double_ma if "双均线" in strategy_type else engine.run_escalator
                res, _, _ = strat_func(s_w, l_w)
                score = evaluate_strategy(res, optimize_metric)
                if score > best_score:
                    best_score = score
                    best_params = {'short': s_w, 'long': l_w}
            my_bar.progress((i + 1) / total)
            
    my_bar.empty()
    return best_params, best_score

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

def plot_period_strategy_chart(df, start_date, end_date, code, strategy_name):
    """绘制选定时间段内的策略走势图（归一化收益对比）"""
    # 筛选时间段数据
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    df_period = df[mask].copy()
    
    if df_period.empty or len(df_period) < 2:
        return None
    
    # 计算归一化收益（从100开始）
    initial_price = df_period['High'].iloc[0]
    df_period['Price_Normalized'] = df_period['High'] / initial_price * 100
    
    # 计算策略收益（考虑信号）
    df_period['Daily_Return'] = df_period['High'].pct_change()
    df_period['Strategy_Daily_Return'] = df_period['Daily_Return'] * df_period['Signal'].shift(1)
    df_period['Strategy_Normalized'] = 100 * (1 + df_period['Strategy_Daily_Return']).cumprod()
    df_period['Strategy_Normalized'].iloc[0] = 100  # 起始点设为100
    
    # 创建子图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.08, row_heights=[0.7, 0.3],
                       subplot_titles=(f"📈 {code} 策略走势对比 ({start_date} ~ {end_date})", "持仓信号"))
    
    # 第一行：归一化价格 vs 策略收益
    fig.add_trace(go.Scatter(x=df_period.index, y=df_period['Price_Normalized'], 
                            name='买入持有', line=dict(color='#9E9E9E', width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_period.index, y=df_period['Strategy_Normalized'], 
                            name='策略收益', line=dict(color='#2196F3', width=2)),
                  row=1, col=1)
    
    # 添加买卖信号点
    buy_signals = df_period[df_period['Position'] == 1]
    sell_signals = df_period[df_period['Position'] == -1]
    
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Strategy_Normalized'], 
                                mode='markers', name='买入',
                                marker=dict(symbol='triangle-up', size=12, color='red')),
                      row=1, col=1)
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Strategy_Normalized'], 
                                mode='markers', name='卖出',
                                marker=dict(symbol='triangle-down', size=12, color='green')),
                      row=1, col=1)
    
    # 第二行：持仓信号
    fig.add_trace(go.Scatter(x=df_period.index, y=df_period['Signal'], 
                            name='持仓', fill='tozeroy',
                            line=dict(color='#4CAF50', width=1),
                            fillcolor='rgba(76,175,80,0.3)'),
                  row=2, col=1)
    
    # 计算区间统计
    price_return = (df_period['Price_Normalized'].iloc[-1] - 100)
    strategy_return = (df_period['Strategy_Normalized'].iloc[-1] - 100)
    excess_return = strategy_return - price_return
    
    # 添加注释
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"买入持有: {price_return:+.2f}%<br>策略收益: {strategy_return:+.2f}%<br>超额收益: {excess_return:+.2f}%",
        showarrow=False, font=dict(size=12),
        align="left", bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#CCCCCC", borderwidth=1
    )
    
    fig.update_yaxes(title_text="归一化收益 (起始=100)", row=1, col=1)
    fig.update_yaxes(title_text="持仓", tickvals=[0, 1], ticktext=['空仓', '持有'], row=2, col=1)
    
    fig.update_layout(height=500, template="plotly_white", hovermode="x unified",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig, {
        'price_return': price_return,
        'strategy_return': strategy_return,
        'excess_return': excess_return,
        'trade_count': len(buy_signals) + len(sell_signals),
        'days': len(df_period)
    }

# --- 7. 主程序 ---
def main():
    st.title("📈 ZC_金银走势追踪")
    
    # 手动刷新按钮
    if st.sidebar.button("🔄 刷新数据"):
        st.cache_data.clear()
        st.rerun()
    
    # 侧边栏配置
    ASSET_OPTIONS = {'AU.SHF': '黄金期货', 'AG.SHF': '白银期货', 'Au9999.SGE': '黄金现货','TL.CFE': '30年国债主连','000905.SHF':'中证500'}
    target_code = st.sidebar.selectbox("选择标的", options=list(ASSET_OPTIONS.keys()), 
                                        format_func=lambda x: ASSET_OPTIONS[x])
    period_mode = st.sidebar.radio("选择周期", ["日线", "周线"], horizontal=True)
    strategy_type = st.sidebar.radio("选择策略", [
        "双均线策略 (Double MA)", 
        "自动扶梯策略 (Escalator)", 
        "R-Breaker 策略",
        "Dual Thrust 策略"
    ])
    
    # 回撤分析开关
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 回撤分析")
    show_drawdown = st.sidebar.checkbox("显示动态回撤", value=True)
    show_trades = st.sidebar.checkbox("显示交易明细", value=True)
    show_equity = st.sidebar.checkbox("显示权益曲线", value=False)
    
    # 新增：时间段策略走势图
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 时间段策略走势")
    show_period_chart = st.sidebar.checkbox("显示时间段策略走势", value=False)
    
    # 新增：策略回测功能
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 策略回测")
    show_backtest = st.sidebar.checkbox("启用完整回测", value=False, 
                                        help="开启后将显示完整的回测分析面板")
    
    if show_backtest:
        bt_initial_capital = st.sidebar.number_input("初始资金", 
                                                     min_value=10000, 
                                                     max_value=10000000, 
                                                     value=100000,
                                                     step=10000,
                                                     help="回测初始资金")
        bt_commission = st.sidebar.slider("手续费率 (‱)", 
                                          min_value=0, 
                                          max_value=50, 
                                          value=3,
                                          help="万分之几") / 10000
        bt_slippage = st.sidebar.slider("滑点 (‱)", 
                                        min_value=0, 
                                        max_value=20, 
                                        value=1,
                                        help="万分之几") / 10000
        bt_position_size = st.sidebar.slider("仓位比例 (%)", 
                                             min_value=10, 
                                             max_value=100, 
                                             value=100,
                                             help="每次交易使用的资金比例") / 100
    else:
        bt_initial_capital = 100000
        bt_commission = 0.0003
        bt_slippage = 0.0001
        bt_position_size = 1.0

    # 加载数据
    df_raw, path = load_csv_data(target_code)
    if df_raw.empty:
        st.error(f"未找到数据文件 {target_code}.csv")
        return

    df_active = resample_data(df_raw, 'W' if "周线" in period_mode else 'D')
    engine = StrategyEngine(df_active)

    # 优化按钮逻辑
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 参数优化")
    
    # 优化指标选择
    optimize_metric = st.sidebar.selectbox(
        "优化目标",
        options=['sharpe', 'return', 'drawdown', 'trade_count'],
        format_func=lambda x: {
            'sharpe': '📈 夏普率 ',
            'return': '💰 总收益率',
            'drawdown': '🛡️ 最小回撤',
            'trade_count': '🔄 交易次数'
        }.get(x, x),
        index=0,
        help="选择参数优化的目标指标"
    )
    
    if st.sidebar.button("🔍 搜索最优参数"):
        best_p, best_score = optimize_parameters(df_active, strategy_type, engine, optimize_metric)
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
        
        # 显示优化结果
        metric_labels = {
            'sharpe': '夏普率',
            'return': '总收益率(%)',
            'drawdown': '最小回撤(%)',
            'trade_count': '交易次数'
        }
        st.toast(f"优化完成！最优{metric_labels[optimize_metric]}: {best_score:.2f}", icon="✅")

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
    
    # 计算策略指标
    current_sharpe = calculate_sharpe_ratio(df_res)
    current_return = calculate_total_return(df_res)

    # 显示指标面板
    last = df_res.iloc[-1]
    
    # 第一行指标
    cols = st.columns(6)
    cols[0].metric("当前最新价", f"{last['Close']:.2f}")
    
    current_status = "多头持有" if last['Signal'] == 1 else "空仓观望"
    cols[1].metric("策略状态", current_status)
    
    signal_count = len(df_res[df_res['Position'] != 0])
    cols[2].metric("交易次数", f"{signal_count} 次")
    
    # 夏普率
    sharpe_display = f"{current_sharpe:.2f}" if current_sharpe != -np.inf else "N/A"
    cols[3].metric("📈 夏普率", sharpe_display)
    
    # 总收益率
    return_display = f"{current_return:+.2f}%" if current_return != -np.inf else "N/A"
    cols[4].metric("💰 总收益率", return_display)
    
    # 最大回撤
    if 'Drawdown' in df_res.columns:
        max_dd = df_res['Drawdown'].min()
        cols[5].metric("最大回撤", f"{max_dd:.2f}%")
    else:
        cols[5].metric("数据更新", last.name.strftime('%Y-%m-%d'))

    # 绘图
    st.plotly_chart(plot_chart(df_res, target_code, l1, l2, strategy_type, period_mode, show_drawdown), 
                   use_container_width=True)
    
    # 显示权益曲线
    if show_equity:
        equity_fig = plot_equity_curve(df_res)
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
    
    # 显示时间段策略走势图
    if show_period_chart:
        st.markdown("---")
        st.subheader("📈 时间段策略走势分析")
        
        # 获取数据的日期范围
        min_date = df_res.index.min().date()
        max_date = df_res.index.max().date()
        
        # 时间段选择
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("开始日期", 
                                       value=max_date - pd.Timedelta(days=365),
                                       min_value=min_date, 
                                       max_value=max_date,
                                       key="period_start")
        with col_date2:
            end_date = st.date_input("结束日期", 
                                     value=max_date,
                                     min_value=min_date, 
                                     max_value=max_date,
                                     key="period_end")
        
        # 快捷时间段按钮
        quick_cols = st.columns(6)
        quick_periods = [
            ("近1月", 30), ("近3月", 90), ("近6月", 180),
            ("近1年", 365), ("近2年", 730), ("全部", None)
        ]
        
        for i, (label, days) in enumerate(quick_periods):
            if quick_cols[i].button(label, key=f"quick_{label}"):
                if days is None:
                    start_date = min_date
                else:
                    start_date = max(min_date, max_date - pd.Timedelta(days=days))
                end_date = max_date
                st.session_state.period_start = start_date
                st.session_state.period_end = end_date
                st.rerun()
        
        # 绘制时间段图表
        if start_date < end_date:
            result = plot_period_strategy_chart(df_res, start_date, end_date, target_code, strategy_type)
            if result:
                period_fig, period_stats = result
                
                # 显示区间统计
                stat_cols = st.columns(4)
                stat_cols[0].metric("买入持有收益", f"{period_stats['price_return']:+.2f}%")
                stat_cols[1].metric("策略收益", f"{period_stats['strategy_return']:+.2f}%",
                                   delta=f"{period_stats['excess_return']:+.2f}%")
                stat_cols[2].metric("区间交易次数", f"{period_stats['trade_count']} 次")
                stat_cols[3].metric("区间天数", f"{period_stats['days']} 天")
                
                st.plotly_chart(period_fig, use_container_width=True)
            else:
                st.warning("所选时间段数据不足，请调整日期范围")
        else:
            st.warning("请确保开始日期早于结束日期")

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

    # 完整回测分析面板
    if show_backtest:
        st.markdown("---")
        st.subheader("🔬 策略回测分析")
        
        # 创建回测引擎并运行回测
        bt_engine = BacktestEngine(
            initial_capital=bt_initial_capital,
            commission_rate=bt_commission,
            slippage=bt_slippage
        )
        
        backtest_result = bt_engine.run_backtest(df_res, position_size=bt_position_size)
        
        if backtest_result:
            metrics = backtest_result['metrics']
            bt_trades_df = backtest_result['trades_df']
            
            # 回测参数概览
            with st.expander("📋 回测参数", expanded=False):
                param_cols = st.columns(4)
                param_cols[0].info(f"初始资金: ¥{bt_initial_capital:,.0f}")
                param_cols[1].info(f"手续费率: {bt_commission*10000:.1f}‱")
                param_cols[2].info(f"滑点: {bt_slippage*10000:.1f}‱")
                param_cols[3].info(f"仓位比例: {bt_position_size*100:.0f}%")
            
            # 核心指标面板
            st.markdown("### 📊 核心业绩指标")
            
            # 第一行：收益指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 总收益率", f"{metrics.get('总收益率(%)', 0):+.2f}%",
                       delta=f"年化 {metrics.get('年化收益率(%)', 0):+.2f}%")
            col2.metric("📈 最终资金", f"¥{metrics.get('最终资金', 0):,.2f}",
                       delta=f"¥{metrics.get('最终资金', 0) - metrics.get('初始资金', 0):+,.2f}")
            col3.metric("🎯 夏普率", f"{metrics.get('夏普率', 0):.2f}")
            col4.metric("📉 最大回撤", f"{metrics.get('最大回撤(%)', 0):.2f}%")
            
            # 第二行：交易指标
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("🔄 交易次数", f"{metrics.get('交易次数', 0)} 次")
            col6.metric("🎲 胜率", f"{metrics.get('胜率(%)', 0):.1f}%")
            col7.metric("💹 盈亏比", f"{metrics.get('盈亏比', 0):.2f}")
            col8.metric("📊 盈利因子", f"{metrics.get('盈利因子', 0):.2f}")
            
            # 第三行：风险指标
            col9, col10, col11, col12 = st.columns(4)
            col9.metric("📊 年化波动率", f"{metrics.get('年化波动率(%)', 0):.2f}%")
            col10.metric("⚖️ Calmar比率", f"{metrics.get('Calmar比率', 0):.2f}")
            col11.metric("💵 总手续费", f"¥{metrics.get('总手续费', 0):,.2f}")
            col12.metric("📅 回测天数", f"{metrics.get('回测天数', 0)} 天")
            
            # 最大回撤区间信息
            st.info(f"📌 最大回撤区间: {metrics.get('最大回撤开始', 'N/A')} ~ {metrics.get('最大回撤结束', 'N/A')}")
            
            # 回测图表
            st.markdown("### 📈 净值曲线与回撤")
            bt_fig = plot_backtest_result(backtest_result, df_res)
            st.plotly_chart(bt_fig, use_container_width=True)
            
            # 月度收益热力图
            st.markdown("### 🗓️ 月度收益分析")
            try:
                monthly_fig = plot_monthly_returns(backtest_result)
                st.plotly_chart(monthly_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"月度收益图表生成失败: {e}")
            
            # 交易分布分析
            if len(bt_trades_df) > 0:
                st.markdown("### 📊 交易分布分析")
                trade_dist_fig = plot_trade_distribution(bt_trades_df)
                if trade_dist_fig:
                    st.plotly_chart(trade_dist_fig, use_container_width=True)
                
                # 详细交易记录
                st.markdown("### 📋 详细交易记录")
                
                # 添加筛选功能
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_win_only = st.checkbox("仅显示盈利交易", key="bt_win_filter")
                with filter_col2:
                    show_lose_only = st.checkbox("仅显示亏损交易", key="bt_lose_filter")
                
                display_bt_trades = bt_trades_df.copy()
                if show_win_only:
                    display_bt_trades = display_bt_trades[display_bt_trades['收益率(%)'] > 0]
                if show_lose_only:
                    display_bt_trades = display_bt_trades[display_bt_trades['收益率(%)'] < 0]
                
                # 显示表格
                def highlight_bt_pnl(val):
                    if isinstance(val, (int, float)):
                        color = 'color: green' if val > 0 else 'color: red' if val < 0 else ''
                        return color
                    return ''
                
                if len(display_bt_trades) > 0:
                    st.dataframe(
                        display_bt_trades.style.applymap(highlight_bt_pnl, subset=['收益率(%)', '盈亏金额']),
                        use_container_width=True,
                        height=400
                    )
                    
                    # 下载回测结果
                    st.markdown("### 📥 导出回测结果")
                    dl_col1, dl_col2 = st.columns(2)
                    
                    with dl_col1:
                        csv_trades = bt_trades_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载交易记录 (CSV)",
                            data=csv_trades,
                            file_name=f"{target_code}_{strategy_type}_backtest_trades.csv",
                            mime="text/csv"
                        )
                    
                    with dl_col2:
                        # 导出完整回测报告
                        report_data = {
                            '回测参数': {
                                '标的': target_code,
                                '策略': strategy_type,
                                '周期': period_mode,
                                '初始资金': bt_initial_capital,
                                '手续费率': f"{bt_commission*10000}‱",
                                '滑点': f"{bt_slippage*10000}‱",
                                '仓位比例': f"{bt_position_size*100}%"
                            },
                            '业绩指标': metrics
                        }
                        import json
                        report_json = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
                        st.download_button(
                            label="📥 下载回测报告 (JSON)",
                            data=report_json.encode('utf-8'),
                            file_name=f"{target_code}_{strategy_type}_backtest_report.json",
                            mime="application/json"
                        )
                else:
                    st.info("没有符合筛选条件的交易记录")
        else:
            st.error("回测执行失败，请检查数据是否完整")

    # 数据预览
    with st.expander("查看原始信号数据"):
        display_cols = ['Open', 'High', 'Low', 'Close', 'Signal', 'Position']
        if "Dual Thrust" in strategy_type:
            display_cols.extend(['Upper_Band', 'Lower_Band', 'Range', 'HH', 'LL'])
        if 'Drawdown' in df_res.columns:
            display_cols.append('Drawdown')
        
        available_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(df_res[available_cols], use_container_width=True)

if __name__ == "__main__":
    main()

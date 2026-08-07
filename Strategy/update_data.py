
import akshare as ak
import pandas as pd
import requests
import os
import time
from datetime import datetime

# --- 配置 ---
CODE_MAP = {
    'AU.SHF': 'au0',       # 上海黄金期货主力连续
    'AG.SHF': 'ag0',       # 上海白银期货主力连续
    'TL.CFE': 'TL0',       # 30年国债期货主力连续
    'Au9999.SGE': 'Au99.99', # 上海黄金交易所现货
    '000905.SHF': '000905',   # 中证500指数
    '932000.CSI': '932000'    # 中证2000指数 (新增)
}

DATA_DIR = "data"


def fetch_csi_index(symbol: str, start_date: str = "20000101") -> pd.DataFrame:
    """
    从中证指数公司官网直接获取指数历史数据。

    背景：ak.index_zh_a_hist() 内部裸调 requests.get() 请求东方财富
    push2his.eastmoney.com，既无浏览器请求头也无 TLS 指纹模拟，
    东方财富服务器对频繁请求做 IP 级反爬（RemoteDisconnected）。
    本函数改用 www.csindex.com.cn 官方 API，无反爬问题，
    同时支持 000905 和 932000，返回完整 OHLCV。

    Args:
        symbol: 指数代码，如 '000905'（中证500）、'932000'（中证2000）
        start_date: 起始日期，格式 YYYYMMDD

    Returns:
        包含 Date/Open/High/Low/Close/Volume 列的 DataFrame

    Raises:
        requests.RequestException: 网络请求失败时抛出
    """
    url = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
    end_date = datetime.now().strftime("%Y%m%d")
    params = {"indexCode": symbol, "startDate": start_date, "endDate": end_date}

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.rename(columns={
        "tradeDate": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tradingVol": "Volume",
    }, inplace=True)

    # 过滤基准日（OHLC 为 None 的行，如 932000 的 2000-01-01 基准行）
    df = df.dropna(subset=["Open", "High", "Low"]).reset_index(drop=True)

    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    return df[[c for c in cols if c in df.columns]].copy()


def fetch_index_data(wind_code: str, ak_code: str, max_retries: int = 3) -> pd.DataFrame:
    """
    多源获取股票指数历史数据，按可靠性依次尝试：
      1. CSI 官网 — 同时支持 000905 和 932000，无反爬（首选）
      2. 新浪财经 — 仅支持 000905，作为备选
      3. 东方财富 — 最后手段，带重试（可能被反爬拦截）

    Args:
        wind_code: Wind 代码，如 '000905.SHF'
        ak_code: 指数代码，如 '000905'
        max_retries: 每个数据源的最大重试次数

    Returns:
        包含 OHLCV 列的 DataFrame；全部失败时返回空 DataFrame
    """
    # --- Source 1: CSI 官网（首选）---
    for attempt in range(max_retries):
        try:
            print(f"   [源1-CSI官网] 第 {attempt + 1}/{max_retries} 次尝试...")
            df = fetch_csi_index(ak_code)
            if not df.empty:
                print(f"   ✅ CSI官网获取成功，共 {len(df)} 条")
                return df
            print(f"   ⚠️ CSI官网返回空数据")
        except Exception as e:
            print(f"   ⚠️ CSI官网失败: {type(e).__name__}: {e}")
        time.sleep(2)

    # --- Source 2: 新浪财经（仅 000905）---
    if wind_code == '000905.SHF':
        for attempt in range(max_retries):
            try:
                print(f"   [源2-新浪] 第 {attempt + 1}/{max_retries} 次尝试...")
                df = ak.stock_zh_index_daily(symbol=f"sh{ak_code}")
                if not df.empty:
                    print(f"   ✅ 新浪获取成功，共 {len(df)} 条")
                    return df
                print(f"   ⚠️ 新浪返回空数据")
            except Exception as e:
                print(f"   ⚠️ 新浪失败: {type(e).__name__}: {e}")
            time.sleep(2)

    # --- Source 3: 东方财富（最后手段）---
    for attempt in range(max_retries):
        try:
            print(f"   [源3-东方财富] 第 {attempt + 1}/{max_retries} 次尝试...")
            df = ak.index_zh_a_hist(symbol=ak_code, period="daily", start_date="19700101")
            if not df.empty:
                print(f"   ✅ 东方财富获取成功，共 {len(df)} 条")
                return df
            print(f"   ⚠️ 东方财富返回空数据")
        except Exception as e:
            print(f"   ⚠️ 东方财富失败: {type(e).__name__}: {e}")
        time.sleep(5)

    return pd.DataFrame()


def update_data_akshare():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 创建数据目录: {DATA_DIR}")

    for wind_code, ak_code in CODE_MAP.items():
        print(f"\n📡 正在获取 {wind_code} (AkShare代码: {ak_code})...")
        
        try:
            df = pd.DataFrame()

            # 1. 处理期货类 (黄金、白银、30年国债)
            if wind_code in ['AU.SHF', 'AG.SHF', 'TL.CFE']:
                df = ak.futures_main_sina(symbol=ak_code)

            # 2. 处理黄金现货
            elif wind_code == 'Au9999.SGE':
                df = ak.spot_hist_sge(symbol=ak_code)
            
            # 3. 处理股票指数 (中证500, 中证2000) — 多源获取，绕过东方财富反爬
            elif wind_code in ['000905.SHF', '932000.CSI']:
                df = fetch_index_data(wind_code, ak_code)
            
            if df.empty:
                print(f"⚠️ {wind_code} 获取到的数据为空，跳过。")
                continue

            # 列名映射 (统一格式)
            rename_map = {
                '日期': 'Date', 'date': 'Date', 'Date': 'Date',
                '收盘价': 'Close', '收盘': 'Close', 'close': 'Close', 'price': 'Close', 'last': 'Close',
                '最高价': 'High', '最高': 'High', 'high': 'High', 'max': 'High',
                '最低价': 'Low', '最低': 'Low', 'low': 'Low', 'min': 'Low',
                '开盘价': 'Open', '开盘': 'Open', 'open': 'Open',
                '成交量': 'Volume', 'vol': 'Volume', 'volume': 'Volume', 'tradingVol': 'Volume'
            }
            
            df.rename(columns=rename_map, inplace=True)

            # 检查必要列
            required_cols = ['Date', 'Close']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                print(f"❌ 严重错误：{wind_code} 缺少必要列: {missing_cols}")
                continue
                
            # 补齐 High/Low
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']

            # 整理 DataFrame
            cols_to_keep = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[cols_to_keep].copy()

            # 日期处理
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # 数值清洗
            for col in df.columns:
                if df[col].dtype == 'object':
                     df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 去除空值
            df.dropna(subset=['Close'], inplace=True)

            # 保存文件
            file_path = os.path.join(DATA_DIR, f"{wind_code}.csv")
            df.to_csv(file_path)
            
            print(f"✅ 成功保存: {file_path}")
            print(f"   数据范围: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   包含列名: {list(df.columns)}")

        except Exception as e:
            print(f"❌ 处理 {wind_code} 时发生异常: {e}")

        # 请求间短暂延迟，避免触发反爬
        time.sleep(1)

    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    update_data_akshare()

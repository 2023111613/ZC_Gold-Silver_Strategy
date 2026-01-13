# 文件名: update_data.py (云端版 - 已新增30年国债)
import akshare as ak
import pandas as pd
import os
from datetime import datetime

# --- 配置 ---
CODE_MAP = {
    'AU.SHF': 'au0',       # 上海黄金期货主力连续
    'AG.SHF': 'ag0',       # 上海白银期货主力连续
    'TL.CFE': 'TL0',       # 30年国债期货主力连续 (新增)
    'Au9999.SGE': 'Au99.99' ,# 上海黄金交易所现货
    '000905.SHF':'中证500'    #中证500 
}

DATA_DIR = "data"

def update_data_akshare():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 创建数据目录: {DATA_DIR}")

    for wind_code, ak_code in CODE_MAP.items():
        print(f"\n📡 正在获取 {wind_code} (AkShare代码: {ak_code})...")
        
        try:
            df = pd.DataFrame()

            if wind_code in ['AU.SHF', 'AG.SHF', 'TL.CFE']:
                df = ak.futures_main_sina(symbol=ak_code)
                # 返回列通常包括：日期, 开盘价, 最高价, 最低价, 收盘价, 成交量, 持仓量

            elif wind_code == 'Au9999.SGE':
                df = ak.spot_hist_sge(symbol=ak_code)
            
            elif wind_code == '000905.SHF':
                # 中证500指数
                df = ak.index_zh_a_hist(symbol="000905", period="daily", start_date="19700101")
                # 返回列通常包括：日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
            
            if df.empty:
                print(f"⚠️ {wind_code} 获取到的数据为空，跳过。")
                continue

            rename_map = {
                '日期': 'Date', 'date': 'Date', 'Date': 'Date',
                '收盘价': 'Close', '收盘': 'Close', 'close': 'Close', 'price': 'Close', 'last': 'Close',
                '最高价': 'High', '最高': 'High', 'high': 'High', 'max': 'High',
                '最低价': 'Low', '最低': 'Low', 'low': 'Low', 'min': 'Low',
                '开盘价': 'Open', '开盘': 'Open', 'open': 'Open',
                '成交量': 'Volume', 'vol': 'Volume', 'volume': 'Volume'
            }
            
            df.rename(columns=rename_map, inplace=True)

            required_cols = ['Date', 'Close', 'High', 'Low']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if 'Date' in missing_cols:
                print(f"❌ 严重错误：{wind_code} 找不到日期列。")
                continue
                
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']

            cols_to_keep = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[cols_to_keep].copy()

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # 数值清洗（处理逗号和非数值类型）
            for col in df.columns:
                if df[col].dtype == 'object':
                     df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 去除空值（如果有）
            df.dropna(subset=['Close'], inplace=True)

            file_path = os.path.join(DATA_DIR, f"{wind_code}.csv")
            df.to_csv(file_path)
            
            print(f"成功保存: {file_path}")
            print(f" 数据范围: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
            print(f" 包含列名: {list(df.columns)}")

        except Exception as e:
            print(f"❌ 处理 {wind_code} 时发生异常: {e}")

    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    update_data_akshare()

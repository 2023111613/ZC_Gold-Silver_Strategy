# 文件名: update_data.py (云端版)
import akshare as ak
import pandas as pd
import os
from datetime import datetime

CODE_MAP = {
    'AU.SHF': 'au0',  
    'AG.SHF': 'ag0',
    'Au9999.SGE': '黄金9999'  # 现货
}

data_dir = "data"

def update_data_akshare():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("开始从 AkShare 获取数据...")

    for wind_code, ak_code in CODE_MAP.items():
        print(f"正在获取 {wind_code} (AkShare代码: {ak_code})...")
        
        try:
            df = pd.DataFrame()
            
            # 1. 期货主力连续
            if wind_code in ['AU.SHF', 'AG.SHF']:
                df = ak.futures_main_sina(symbol=ak_code)

                # --- 关键：把高低价一起取出来 ---
                # 视你本地 akshare 版本，列名可能是中文或英文，两种都兼容一下
                col_map = {
                    '日期': 'Date', 'date': 'Date',
                    '收盘价': 'Close', 'close': 'Close',
                    '最高价': 'High', 'high': 'High',
                    '最低价': 'Low', 'low': 'Low',
                }
                df = df.rename(columns=col_map)

                # 只保留我们需要的几列
                df = df[['Date', 'Close', 'High', 'Low']].copy()

            # 2. 黄金现货 Au9999.SGE
            elif wind_code == 'Au9999.SGE':
                df = ak.spot_hist_sge(symbol=ak_code)

                # 同样做一遍列名映射
                col_map = {
                    '日期': 'Date', 'date': 'Date',
                    '收盘': 'Close', 'close': 'Close',
                    '最高': 'High', 'high': 'High',
                    '最低': 'Low', 'low': 'Low',
                }
                df = df.rename(columns=col_map)
                # 有些版本可能只有 close，这里尽量多取
                use_cols = [c for c in ['Date', 'Close', 'High', 'Low'] if c in df.columns]
                df = df[use_cols].copy()

            # --- 统一格式处理 ---
            if not df.empty:
                # 时间索引
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # 数值化
                for col in ['Close', 'High', 'Low']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                file_path = os.path.join(data_dir, f"{wind_code}.csv")
                df.to_csv(file_path)
                print(f"✅ 保存成功: {file_path}，列: {list(df.columns)}")
            else:
                print(f"⚠️ {wind_code} 数据为空")

        except Exception as e:
            print(f"❌ 获取 {wind_code} 失败: {e}")

if __name__ == "__main__":
    update_data_akshare()

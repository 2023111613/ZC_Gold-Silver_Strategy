# 文件名: update_data.py (云端版)
import akshare as ak
import pandas as pd
import os
from datetime import datetime

# 建立 Wind 代码到 AkShare 代码的映射
# AkShare 黄金期货主力连续通常叫 "au0", 白银 "ag0"
CODE_MAP = {
    'AU.SHF': 'au0',  
    'AG.SHF': 'ag0',
    'Au9999.SGE': '黄金9999' # 现货
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
            
            # 1. 处理期货数据 (au0, ag0)
            if wind_code in ['AU.SHF', 'AG.SHF']:
                # 获取期货主力连续数据
                df = ak.futures_main_sina(symbol=ak_code)
                # 清洗数据，AkShare返回列名通常是中文
                df = df[['日期', '收盘价']].copy()
                df.columns = ['Date', 'Close']
            
            # 2. 处理黄金现货数据
            elif wind_code == 'Au9999.SGE':
                # 获取现货数据
                df = ak.spot_hist_sge(symbol=ak_code)
                df = df[['date', 'close']].copy()
                df.columns = ['Date', 'Close']

            # 统一格式处理
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df['Close'] = pd.to_numeric(df['Close'])
                
                # 保存
                file_path = os.path.join(data_dir, f"{wind_code}.csv")
                df.to_csv(file_path)
                print(f"✅ 保存成功: {file_path}")
            else:
                print(f"⚠️ {wind_code} 数据为空")

        except Exception as e:
            print(f"❌ 获取 {wind_code} 失败: {e}")

if __name__ == "__main__":
    update_data_akshare()

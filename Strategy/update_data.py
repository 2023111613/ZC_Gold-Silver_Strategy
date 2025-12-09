# 文件名: update_data.py (云端版)
import akshare as ak
import pandas as pd
import os
from datetime import datetime

# --- 配置 ---
CODE_MAP = {
    'AU.SHF': 'au0',       # 上海黄金期货主力连续
    'AG.SHF': 'ag0',       # 上海白银期货主力连续
    'Au9999.SGE': 'Au99.99' # 上海黄金交易所现货
}

DATA_DIR = "data"

def update_data_akshare():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 创建数据目录: {DATA_DIR}")

    print("🚀 开始从 AkShare 获取数据...")

    for wind_code, ak_code in CODE_MAP.items():
        print(f"\n📡 正在获取 {wind_code} (AkShare代码: {ak_code})...")
        
        try:
            df = pd.DataFrame()
            
            # ---------------------------
            # 1. 期货数据 (新浪财经接口)
            # ---------------------------
            if wind_code in ['AU.SHF', 'AG.SHF']:
                df = ak.futures_main_sina(symbol=ak_code)
                # 典型返回列: 日期, 开盘价, 最高价, 最低价, 收盘价, 成交量, 持仓量

            # ---------------------------
            # 2. 现货数据 (上海黄金交易所接口)
            # ---------------------------
            elif wind_code == 'Au9999.SGE':
                # 注意：spot_hist_sge 接口有时不稳定，如果报错，需检查 akshare 版本
                df = ak.spot_hist_sge(symbol=ak_code)
            
            if df.empty:
                print(f"⚠️ {wind_code} 获取到的数据为空，跳过。")
                continue

            # ---------------------------
            # 3. 统一列名清洗 (核心步骤)
            # ---------------------------
            # 建立一个超级映射表，兼容中文、英文、大小写
            rename_map = {
                # 日期
                '日期': 'Date', 'date': 'Date', 'Date': 'Date',
                # 收盘
                '收盘价': 'Close', '收盘': 'Close', 'close': 'Close', 'price': 'Close', 'last': 'Close',
                # 最高
                '最高价': 'High', '最高': 'High', 'high': 'High', 'max': 'High',
                # 最低
                '最低价': 'Low', '最低': 'Low', 'low': 'Low', 'min': 'Low',
                # 开盘
                '开盘价': 'Open', '开盘': 'Open', 'open': 'Open',
                # 量
                '成交量': 'Volume', 'vol': 'Volume', 'volume': 'Volume'
            }
            
            # 先把列名去空格并转小写(辅助匹配)，但为了映射表生效，我们直接重命名匹配到的
            df.rename(columns=rename_map, inplace=True)

            # ---------------------------
            # 4. 确保必要列存在
            # ---------------------------
            required_cols = ['Date', 'Close', 'High', 'Low']
            
            # 检查缺失列
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if 'Date' in missing_cols:
                print("❌ 严重错误：找不到日期列，无法处理。")
                continue
                
            # 如果缺少 High/Low (比如某些现货源只有收盘价)，用 Close 填充，防止策略报错
            if 'High' not in df.columns:
                print("⚠️ 警告：缺失 'High' 列，使用 'Close' 填充")
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                print("⚠️ 警告：缺失 'Low' 列，使用 'Close' 填充")
                df['Low'] = df['Close']

            # 只保留需要的列
            cols_to_keep = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            df = df[cols_to_keep].copy()

            # ---------------------------
            # 5. 格式转换与保存
            # ---------------------------
            # 处理时间
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # 处理数值 (防止千分位字符串 '1,234.00' 导致报错)
            for col in df.columns:
                # 尝试转为字符串，去掉逗号，再转数字
                if df[col].dtype == 'object':
                     df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 保存 CSV
            file_path = os.path.join(DATA_DIR, f"{wind_code}.csv")
            df.to_csv(file_path)
            
            print(f"✅ 成功保存: {file_path}")
            print(f"   📊 数据范围: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   📝 包含列名: {list(df.columns)}")

        except Exception as e:
            print(f"❌ 处理 {wind_code} 时发生异常: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 所有任务完成！")

if __name__ == "__main__":
    # 确保安装了 akshare: pip install akshare --upgrade
    update_data_akshare()


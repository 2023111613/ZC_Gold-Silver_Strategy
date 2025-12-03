# 文件名: update_data.py
import pandas as pd
from datetime import datetime, timedelta
from WindPy import w
import os

# 配置
targets = ['AU.SHF', 'AG.SHF', 'Au9999.SGE']
start_date = "2020-01-01" # 数据起点
end_date = datetime.now().strftime("%Y-%m-%d")
data_dir = "data" # 数据保存目录

def update_local_data():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print("正在连接 Wind API...")
    res = w.start()
    if res.ErrorCode != 0:
        print("Wind 启动失败")
        return

    for code in targets:
        print(f"正在下载 {code} ...")
        # PriceAdj=F 前复权
        wind_data = w.wsd(code, "close", start_date, end_date, "PriceAdj=F")
        
        if wind_data.ErrorCode == 0:
            df = pd.DataFrame(wind_data.Data[0], index=wind_data.Times, columns=['Close'])
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            # 保存到 data 文件夹
            file_path = os.path.join(data_dir, f"{code}.csv")
            df.to_csv(file_path)
            print(f"成功保存: {file_path}")
        else:
            print(f"{code} 下载失败: {wind_data.ErrorCode}")

    print("所有数据更新完毕！请执行 Git 推送。")

if __name__ == "__main__":
    update_local_data()

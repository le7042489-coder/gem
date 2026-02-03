import os
import sys
import numpy as np
import yaml

# 确保能引用到当前目录下的模块
sys.path.append(os.getcwd())

from ecg_plot import ecg_plot

# 1. 读取项目自带的配置文件
# 必须使用 config.yaml，因为 ecg_plot 内部依赖这个字典结构
config_path = "config.yaml"
if not os.path.exists(config_path):
    print(f"Error: 找不到 {config_path}，请确保你在 ecg-image-generator 目录下运行")
    sys.exit(1)

with open(config_path, 'r') as f:
    configs = yaml.safe_load(f)

# 2. 构造符合要求的伪数据
# 作者的代码要求 ecg 数据是一个字典：{'I': [...], 'II': [...], ...}
lead_names = configs['leadNames_12']  # 获取12导联名称
sample_rate = 500
duration = 10  # 10秒
num_samples = sample_rate * duration

dummy_ecg = {}
print("正在生成模拟信号...")
for lead in lead_names:
    # 随机生成一些类似心电图的起伏数据
    dummy_ecg[lead] = np.random.randn(num_samples) * 0.5 

# 3. 设置输出
output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)
rec_file_name = "test_ecg" # 最终会生成 test_output/test_ecg.png

# 4. 调用绘图函数 (严格匹配原作者的参数定义)
print("正在调用 ecg_plot 绘图...")
try:
    ecg_plot(
        ecg=dummy_ecg, 
        configs=configs, 
        sample_rate=sample_rate, 
        columns=4, 
        rec_file_name=rec_file_name, 
        output_dir=output_dir, 
        resolution=200, 
        pad_inches=0, 
        lead_index=lead_names, 
        full_mode='II', 
        store_text_bbox=False, 
        full_header_file="",  # 传空字符串，只要下面的 print_txt=False 就不会报错
        show_lead_name=True,
        show_grid=True,
        style='bw',           # 'bw' 是黑白，想要彩色可以删掉这行(默认彩色)
        print_txt=False       # 关键：设为 False 就不需要真实的 header 文件
    )
    print(f"✅ 成功！图片已保存到: {os.path.abspath(os.path.join(output_dir, rec_file_name + '.png'))}")

except Exception as e:
    print(f"❌ 运行出错: {e}")
    import traceback
    traceback.print_exc()

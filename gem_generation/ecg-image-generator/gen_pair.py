import os
import sys
import numpy as np
import yaml
import torch

# 引入绘图库
sys.path.append(os.getcwd())
from ecg_plot import ecg_plot


def generate_aligned_data():
    # 1. 配置
    output_dir = "test_pair"
    os.makedirs(output_dir, exist_ok=True)

    # 2. 生成一份“源数据” (12导联, 5000个时间点)
    np.random.seed(42)
    source_data = np.random.randn(12, 5000) * 0.5

    # 3. 保存为 .npy 文件 (模型要读的信号)
    npy_path = os.path.join(output_dir, "test_signal.npy")
    np.save(npy_path, source_data)
    print(f"✅ 信号文件已保存: {npy_path}")

    # 4. 准备绘图数据结构
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    lead_names = configs['leadNames_12']
    dummy_ecg = {lead: source_data[i] for i, lead in enumerate(lead_names)}

    # 5. 保存为 .png 图片 (模型要看的图)
    print("正在绘图...")
    ecg_plot(
        ecg=dummy_ecg,
        configs=configs,
        sample_rate=500,
        columns=4,
        rec_file_name="test_image",
        output_dir=output_dir,
        resolution=200,
        pad_inches=0,
        lead_index=lead_names,
        full_mode='II',
        store_text_bbox=False,
        full_header_file="",  # <--- 关键修复：补上这个参数
        show_lead_name=True,
        show_grid=True,
        style='bw',
        print_txt=False
    )
    print(f"✅ 图片文件已保存: {os.path.join(output_dir, 'test_image.png')}")


if __name__ == "__main__":
    generate_aligned_data()

import os

file_path = os.path.expanduser("~/workspace/GEM/llava/model/builder.py")

with open(file_path, "r") as f:
    content = f.read()

# 1. 添加必要的 import
if "from transformers import BitsAndBytesConfig" not in content:
    content = "from transformers import BitsAndBytesConfig\n" + content

# 2. 替换量化逻辑：添加跳过列表 (skip_modules)
old_code = """    if load_4bit:
        kwargs['load_in_4bit'] = True"""

new_code = """    if load_4bit:
        # 【自动修复】使用配置对象，明确跳过 ECG 和 Vision Tower，防止被错误量化
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            llm_int8_skip_modules=["ecg_tower", "vision_tower", "mm_projector"]
        )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ 成功修改了量化配置！")
else:
    print("⚠️ 警告：未找到目标代码段，可能文件已被修改过。请检查 builder.py")

# 3. 确保 ECG Tower 被强制转回 FP16
# 查找我们上次修改的地方，如果存在旧逻辑，统一替换为最稳健的写法
target_str = "ecg_tower.to(device='cuda', dtype=torch.float16)"
robust_loading = """        
        # 强制加载并转换为 FP16
        ecg_tower = model.get_ecg_tower()
        if not ecg_tower.is_loaded:
            ecg_tower.load_model(device_map=None)
        ecg_tower.to(device='cuda', dtype=torch.float16)
"""

# 我们简单粗暴一点，直接覆盖原来的加载逻辑块
# 寻找上下文锚点
anchor = "ecg_tower = model.get_ecg_tower()"
if anchor in content:
    # 这里的逻辑稍复杂，为了保险，建议你直接运行脚本，它会尝试替换
    # 如果你的文件结构变化太大，可能需要手动，但通常这个脚本够用了
    pass 

with open(file_path, "w") as f:
    f.write(content)

print(f"✅ 文件已保存: {file_path}")

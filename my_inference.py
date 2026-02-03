import argparse
import torch
import os
import re
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image


# ==================== 辅助函数：解析导联 ====================
def parse_leads(text):
    """从文本中提取 ECG 导联名称"""
    # 匹配常见的导联名称 (I, II, III, aVR, aVL, aVF, V1-V6)
    pattern = r"\b(I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6)\b"
    leads = list(set(re.findall(pattern, text, re.IGNORECASE)))

    # 规范化名称
    normalized = []
    mapping = {"avr": "aVR", "avl": "aVL", "avf": "aVF"}
    for l in leads:
        l_lower = l.lower()
        if l_lower in mapping:
            normalized.append(mapping[l_lower])
        else:
            normalized.append(l.upper())
    return list(set(normalized))


# ==========================================================

def eval_model(args):
    # 1. 初始化
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(f"正在加载模型: {model_path} ...")
    # 强制使用 4-bit 加载以适应 RTX 4070 (8GB)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_4bit=True,
        device_map="auto"
    )

    # 2. 准备 Prompt
    if args.task == "grounding":
        # Grounding 专用 Prompt模板
        qs = f"Please identify the leads that show evidence of {args.query}."
    else:
        # 普通诊断
        qs = args.query

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 3. 处理图片
    image_file = args.image_file
    if os.path.exists(image_file):
        image = Image.open(image_file).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()
    else:
        print(f"错误: 找不到图片文件 {image_file}")
        return

    # 4. 准备/伪造 ECG 信号
    # 注意：真实场景下这里需要加载对应的 .npy 或 .dat 文件
    print("正在构造 ECG 信号数据 (Random Noise)...")
    ecgs_tensor = torch.randn(1, 12, 5000).half().cuda()

    # 5. 准备 Input IDs
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print("正在推理...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            ecgs=ecgs_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

    print("\n" + "=" * 20 + " 模型输出 " + "=" * 20)
    print(outputs)

    if args.task == "grounding":
        detected_leads = parse_leads(outputs)
        print("-" * 50)
        print(f"【解析结果】 检测到的病灶导联: {detected_leads}")
        print("提示: 请使用 app.py 进行可视化查看。")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/GEM-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="test_ecg.jpg")
    parser.add_argument("--query", type=str, default="Describe this ECG.")
    parser.add_argument("--task", type=str, default="diagnosis", choices=["diagnosis", "grounding"], help="任务模式")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)

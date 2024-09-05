import os
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc
from PIL import Image
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"使用设备: {device}")
    return device

device = get_device()

def get_model_path():
    local_model_path = os.path.join("model", "Qwen2-VL-7B-Instruct")
    if os.path.exists(local_model_path):
        logger.info(f"使用本地模型: {local_model_path}")
        return local_model_path
    else:
        logger.info("本地模型不存在，使用Hugging Face模型")
        return "Qwen/Qwen2-VL-7B-Instruct"

try:
    logger.info("开始加载模型和处理器")
    model_path = get_model_path()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("模型和处理器加载完成")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    raise

def process_input(image_data, video_path, instruction):
    logger.info("开始处理输入")
    try:
        messages = [{"role": "user", "content": []}]
        
        # 处理图片
        if image_data:
            logger.info("处理图片")
            for item in image_data:
                if isinstance(item, tuple) and len(item) > 0:
                    image_path = item[0]
                    try:
                        img = Image.open(image_path)
                        messages[0]["content"].append({"type": "image", "image": img})
                        logger.info(f"成功加载图片: {image_path}")
                    except Exception as e:
                        logger.error(f"无法加载图片 {image_path}: {str(e)}")
        
        # 处理视频
        if video_path:
            logger.info(f"添加视频路径: {video_path}")
            messages[0]["content"].append({
                "type": "video",
                "video": f"file://{video_path}",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            })
        
        if not messages[0]["content"]:
            logger.warning("没有成功加载任何图片或视频")
            return "请上传至少一张有效的图片或一个有效的视频。"
        
        messages[0]["content"].append({"type": "text", "text": instruction})
        logger.info(f"使用指令: {instruction}")

        logger.info("开始应用聊天模板")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logger.info("处理视觉信息")
        image_inputs, video_inputs = process_vision_info(messages)
        logger.info("准备模型输入")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        logger.info("开始生成输出")
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        combined_output = " ".join(output_texts)
        logger.info("输出生成完成")
        
        # 清理内存
        del inputs, generated_ids, generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("内存清理完成")
        
        return combined_output
    except Exception as e:
        logger.error(f"处理输入时出错: {str(e)}", exc_info=True)
        return f"处理输入时出错: {str(e)}\n错误类型: {type(e).__name__}"

with gr.Blocks() as iface:
    gr.Markdown("# 基于Qwen2-VL的多模态处理器（支持图片、视频和自定义指令）")
    gr.Markdown("上传图片或视频，并输入指令，获取由Qwen2-VL模型生成的分析结果。本应用支持 CUDA, MPS 和 CPU。")
    
    with gr.Row():
        with gr.Column():
            gallery = gr.Gallery(
                label="上传图片", 
                show_label=True, 
                elem_id="gallery",
                columns=2,
                rows=2,
                object_fit="contain", 
                height="auto",
                type="filepath"
            )
        with gr.Column():
            video = gr.Video(label="上传视频")
        
    instruction = gr.Textbox(
        label="输入指令", 
        placeholder="例如：请描述这些图片或视频的内容。", 
        value="请描述这些图片或视频的内容。"
    )
    
    submit_btn = gr.Button("提交")
    output = gr.Textbox(label="模型输出")

    submit_btn.click(
        fn=process_input,
        inputs=[gallery, video, instruction],
        outputs=output
    )

logger.info("启动Gradio界面")
iface.launch()
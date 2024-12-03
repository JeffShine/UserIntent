import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
def build_llm(path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16, device_map='balanced_low_0',attn_implementation="flash_attention_2",
        )

    #processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #processor.apply_chat_template
    return model
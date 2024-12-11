import torch
import torch.nn as nn
from builder.LLM_builder import build_llm
from builder.ocr_builder import build_ocr
from transformers import  AutoProcessor
from qwen_vl_utils import process_vision_info
from abc import ABC, abstractmethod



class InferModel_single(nn.Module):
    def __init__(self,path_llm='/home/pubw/proj/Qwen2-VL-7B-Instruct'):
        super().__init__()
        self.llm=build_llm(path_llm)
        self.processor = AutoProcessor.from_pretrained("/home/pubw/proj/Qwen2-VL-7B-Instruct")


    # messages:[messages1, messages2,]
    @torch.no_grad()
    def infer_llm(self,messages,llm_size='7b'):
        
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )       
        inputs = inputs.to("cuda")
        if llm_size=='7b':
            output_ids = self.llm_7b.generate(**inputs, max_new_tokens=128)
        elif llm_size=='2b':
            output_ids = self.llm_2b.generate(**inputs, max_new_tokens=128)
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text
    
    
    @abstractmethod
    def forward(self,messages):
        pass
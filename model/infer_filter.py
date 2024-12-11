import re
from model.inferModel_base import InferModel_base

photo_template = """{0} 请你对图片进行总结，图片来自于网购手机应用界面"""

base_template="""Picture {0}: <image>\n"""

qa_template1 = """{0}用户提问了这些内容：[{1}],客服回复了这些内容：[{2}], 请你基于用户提出的问题, 对以下OCR列表进行筛选, 只保留与对话有关的元素, 只输出筛选后的OCR列表，不要解释：\n{3}"""
# qa_template2 = """{0}用户提问了这些内容：[{1}],客服回复了这些内容：[{2}], 请你基于用户提出的问题, 对以下OCR列表进行筛选, 只保留与对话有关的元素, 只输出筛选后的OCR列表，不要解释：\n{3}"""

# qa = """用户提问了这些内容：[{0}],客服回复了这些内容：[{1}]\n"""
# image_caption_template="""需要注意的是本应用是淘宝，<image> 一个可能的参考描述是{0} """

def split_conv(conv):
    user_dialogues = []
    service_dialogues=[]
    # 使用 re.split 按照关键点切分字符串
    lines = conv.split("\n")
    for line in lines:
        if line.startswith("用户:"):
            # 提取用户的内容
            #content = line[3:].strip()
            user_dialogues.append(line)
        elif line.startswith("客服:"):
            service_dialogues.append(line)
    
    return user_dialogues,service_dialogues

class infer_filter(InferModel_base):
    
    def forward(self,batch):
        messages_2b=[]
        for data_item in batch:
            img_list=data_item['image']
            ocr_result=data_item['ocr']
            item_messages = []
            base_prompt_list=[]#多图片情况
            for i,img_item in enumerate(img_list):
                base_prompt=base_template.format(i+1)
                base_prompt_list.append(base_prompt)
                image_message = {
                    "type": "image",
                    "image": img_item  
                }
                item_messages.append(image_message)
            img_base=" ".join(base_prompt_list)
            if 'Picture 1' in data_item['instruction'] :
                summary_prompt=photo_template.format(img_base)
            else:
                sp_conv=split_conv(data_item['instruction'])
                summary_prompt=qa_template1.format(img_base,sp_conv[0],sp_conv[1],ocr_result)
        
            # print(summary_prompt)

            item_messages.append({
                "type": "text",
                "text": summary_prompt
            })

            messages_2b.append([{
                "role": "user",
                "content": item_messages
            }])
            
        # print(messages_2b)
        output=self.infer_llm(messages_2b,'7b')
        # print(output)
        return output
        
        

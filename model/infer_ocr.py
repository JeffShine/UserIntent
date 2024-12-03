from model.inferModel_base import InferModel_base
from builder.ocr_builder import build_ocr, extract_ocr_info
ocr_template = """
     Picture <image> {0} 中的文字信息:含有的文本内容中出现频率最高的是{1},最上方的文字是{2},最下方的文字是{3};
    """
taobao_template = """
    Picture {0}: <image> 请你对图片进行总结，图片来自于网购手机应用界面，如果是淘宝或其子页面（菜鸟裹裹等）的软件界面请告诉我可能是哪个软件，请你告诉我最可能是什么界面，"""
base_template="""Picture {0}: <image>\n"""
caption_template = """{0}请你对图片进行总结"""

image_caption_template="""<image> 描述了{0} """

class infer_ocr(InferModel_base):

    def __init__(self,path_2b='/home/pubw/proj/Qwen2-VL-2B-Instruct',path_7b='/home/pubw/proj/Qwen2-VL-7B-Instruct'):
        super().__init__(path_2b,path_7b)
        self.ocr=build_ocr()
    def get_ocr(self,img_path):
        result = self.ocr.ocr(img_path, cls=False)
        return result
    
    def forward(self,data_list):
        messages_2b=[]
        #caption
        for data_item in data_list:
            img_list=data_item['image']
            ocr_prompt=[]
            image_messages = []
            
            for i,img_item in enumerate(img_list):
                if 'Picture 1' in data_item['instruction']:
                    #ocr_res=self.get_ocr(img_item)
                    #if ocr_res[0] != None:
                        #ocr_info=extract_ocr_info(ocr_res[0])
                        #prompt = ocr_template.format(i+1,ocr_info[0], ocr_info[2], ocr_info[3])
                        #ocr_prompt.append(prompt)
                    prompt = ocr_template.format(i+1)
                    ocr_prompt.append(prompt)
                base_prompt=base_template.format(i+1)

                image_message = {
                    "type": "image",
                    "image": img_item  
                }
                image_messages.append(image_message)
            if 'Picture 1' in data_item['instruction'] :
                ocr_info_sum=" ".join(ocr_prompt)
                caption_prompt=taobao_template.format(ocr_info_sum)
            else:
                caption_prompt=base_prompt
             
            image_messages.append({
                "type": "text",
                "text": caption_prompt
            })
        #形成batch
            messages_2b.append([{
                "role": "user",
                "content": image_messages
            }])
        #print(messages_2b)
        image_caption_outputs=self.infer_llm(messages_2b,'2b')
        #QA
        messages_7b=[]
        for data_item in data_list:
            img_list=data_item['image']
            ocr_prompt=[]
            image_messages = []
            image_captions=[]
            for i,img_item in enumerate(img_list):
                image_message = {
                    "type": "image",
                    "image": img_item  
                }
                image_captions.append(image_caption_template.format(image_caption_outputs[i]))
                image_messages.append(image_message)
            
            instruction=[]
            split_instrution=data_item['instruction'].split('<image>')
            for i,sub_ins in enumerate(split_instrution):
                instruction.append(sub_ins)
                if i == len(split_instrution)-2:
                    instruction.append(image_captions[i])
                else:
                    instruction.append("<image>")

            
            image_messages.append({
                "type": "text",
                "text": " ".join(instruction)
            })
        #形成batch
            messages_7b.append([{
                "role": "user",
                "content": image_messages
            }])
        print(messages_2b)
        print(messages_7b)
        output=self.infer_llm(messages_7b,'7b')
        return output
        
        

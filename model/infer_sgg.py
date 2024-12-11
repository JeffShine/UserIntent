import re
from model.inferModel_single import InferModel_single
from builder.ocr_builder import extract_ocr_info

photo_template = """{0} 请你对图片进行总结，图片来自于网购手机应用界面，请你告诉我最可能是什么界面，"""

base_template="""Picture {0}: <image>\n"""

qa_template = """{0}用户提问了这些内容：[{1}],客服回复了这些内容：[{2}], 请你短语总结用户提出的问题，客服有助于解决问题的回复"""

qa = """用户提问了这些内容：[{0}],客服回复了这些内容：[{1}]\n"""
image_caption_template="""需要注意的是本应用是淘宝，<image> 描述了{0} """

task2_label='''[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]'''



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
class infer_sgg(InferModel_single):
    

    def set_attribute(self):
        pass

    def generate_sgg(self,batch):
        pass

    def forward(self,batch):
        messages_2b=[]
        messages_7b=[]
        #caption
        for data_item in batch:
            img_list=data_item['image']
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
                summary_prompt=qa_template.format(img_base,sp_conv[0],sp_conv[1])

            
            item_messages.append({
                "type": "text",
                "text": summary_prompt
            })


            messages_2b.append([{
                "role": "user",
                "content": item_messages
            }])
            
        #print(messages_2b)
        image_caption_outputs=self.infer_llm(messages_2b,'2b')
        
        #可以用caption做一个RAG

        #QA
        
        for sub_i,data_item in enumerate(batch):
            item_messages=[]
            instruction=[]
            
            for i,img_item in enumerate(img_list):
                base_prompt=base_template.format(i+1)
                base_prompt_list.append(base_prompt)
                image_message = {
                    "type": "image",
                    "image": img_item  
                }
                item_messages.append(image_message)
            
           
            if 'Picture 1' in data_item['instruction'] :
                image_caption=image_caption_template.format(image_caption_outputs[sub_i])
                split_instrution=data_item['instruction'].split('<image>')
                for i,sub_ins in enumerate(split_instrution):
                    instruction.append(sub_ins)
                    if i == len(split_instrution)-2:
                        instruction.append('\n'+image_caption+'\n')
                    else:
                        instruction.append("<image>")
            else:
                sp_conv=split_conv(data_item['instruction'])
                instruction.append('你是一个电商客服专家，请根据用户与客服的对话总结以及用户的图片输入判断用户的意图分类标签。\n 初步总结如下可以参考\n')
                instruction.append(qa.format(sp_conv[0],sp_conv[1]))
                instruction.append(image_caption_outputs[sub_i]+'\n')
                instruction.append("""请直接只输出一个分类标签结果，不需要其他多余的话。以下是可以输出的分类标签：[\"反馈密封性不好\",\"是否好用\",\"是否会生锈\",\"排水方式\",\"包装区别\",\"发货数量\",\"反馈用后症状\",\"商品材质\",\"功效功能\",\"是否易褪色\",\"适用季节\",\"能否调光\",\"版本款型区别\",\"单品推荐\",\"用法用量\",\"控制方式\",\"上市时间\",\"商品规格\",\"信号情况\",\"养护方法\",\"套装推荐\",\"何时上货\",\"气泡\"]\n""")
                

            
            item_messages.append({
                "type": "text",
                "text": " ".join(instruction)
            })
        #形成batch
            messages_7b.append([{
                "role": "user",
                "content": item_messages
            }])
        
        output=self.infer_llm(messages_7b,'7b')
        print(image_caption_outputs,output)
        return output
        
        

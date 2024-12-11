import re
from model.inferModel_base import InferModel_base
base_template="""Picture {0}: <image>\n"""

photo_class = """[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]"""
photo_template = """{0}图片来自于网购手机应用界面, 请你基于场景分类任务Task="{1}", 对list="{2}"进行筛选, 保留最有用的元素, 只输出筛选后的list, 不做其他任何解释."""

qa_class = """["反馈密封性不好","是否好用","是否会生锈","排水方式","包装区别","发货数量","反馈用后症状","商品材质","功效功能","是否易褪色","适用季节","能否调光","版本款型区别","单品推荐","用法用量","控制方式","上市时间","商品规格","信号情况","养护方法","套装推荐","何时上货","气泡"]"""
qa_template1 = """{0}用户提问了这些内容：[{1}], 请你基于用户提出的问题和用户意图分类任务Task="{2}", 对list="{3}"进行筛选, 保留最有用的元素, 只输出筛选后的list, 不做其他任何解释."""
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
            ocr_result=data_item['ocr_txts']
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
                summary_prompt=photo_template.format(img_base,photo_class,ocr_result)
            else:
                sp_conv=split_conv(data_item['instruction'])
                summary_prompt=qa_template1.format(img_base,sp_conv[0],qa_class,ocr_result)
        
            print(summary_prompt)

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
        
        

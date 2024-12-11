import json
import os
from tqdm import tqdm
import shutil


json_file = {
    "train_task1": '/home/jch/data/www25/train/train_task1.json',
    "train_task2": '/home/jch/data/www25/train/train_task2.json',
}

img_dir = '/home/jch/data/www25/train/images/'
out_dir = '/home/jch/projects/UserIntent/data_analyze/train_images'

img_dict_task1 = {'反馈密封性不好': [],
 '是否好用': [],
 '是否会生锈': [],
 '排水方式': [],
 '包装区别': [],
 '发货数量': [],
 '反馈用后症状': [],
 '商品材质': [],
 '功效功能': [],
 '是否易褪色': [],
 '适用季节': [],
 '能否调光': [],
 '版本款型区别': [],
 '单品推荐': [],
 '用法用量': [],
 '控制方式': [],
 '上市时间': [],
 '商品规格': [],
 '信号情况': [],
 '养护方法': [],
 '套装推荐': [],
 '何时上货': [],
 '气泡': []}

img_dict_task2 = {'实物拍摄(含售后)': [],
 '商品分类选项': [],
 '商品头图': [],
 '商品详情页截图': [],
 '下单过程中出现异常（显示购买失败浮窗）': [],
 '订单详情页面': [],
 '支付页面': [],
 '消费者与客服聊天页面': [],
 '评论区截图页面': [],
 '物流页面-物流列表页面': [],
 '物流页面-物流跟踪页面': [],
 '物流页面-物流异常页面': [],
 '退款页面': [],
 '退货页面': [],
 '换货页面': [],
 '购物车页面': [],
 '店铺页面': [],
 '活动页面': [],
 '优惠券领取页面': [],
 '账单/账户页面': [],
 '个人信息页面': [],
 '投诉举报页面': [],
 '平台介入页面': [],
 '外部APP截图': [],
 '其他类别图片': []}


with open(json_file["train_task1"], 'r', encoding='utf-8') as file:
    data = json.load(file)
for item in data:
    label = item['output']
    img_dict_task1[label].append((item['image'],item['id']))

with open(json_file["train_task2"], 'r', encoding='utf-8') as file:
    data = json.load(file)
for item in data:
    label = item['output']
    img_dict_task2[label].append((item['image'],item['id']))

# 创建主目录
os.makedirs(out_dir, exist_ok=True)

from PIL import Image
import numpy as np

def concat_images(image_paths):
    """水平拼接多张图片"""
    if not image_paths:
        raise ValueError("Empty image paths list")
    
    # 过滤出存在的图片路径
    valid_paths = [path for path in image_paths if os.path.exists(path)]
    if not valid_paths:
        raise ValueError("No valid image paths found")
    
    images = [Image.open(path) for path in valid_paths]
    heights = [img.size[1] for img in images]
    min_height = min(heights)
    images = [img.resize((int(img.size[0] * min_height / img.size[1]), min_height)) 
             for img in images]
    
    total_width = sum(img.size[0] for img in images)
    new_img = Image.new('RGB', (total_width, min_height))
    
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    return new_img

# 处理task1的图片
task1_dir = os.path.join(out_dir, 'task1')
os.makedirs(task1_dir, exist_ok=True)

for label, img_lists in tqdm(img_dict_task1.items(), desc='Processing Task1'):
    label_dir = os.path.join(task1_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    for i, (img_list, id) in enumerate(img_lists):
        if len(img_list) == 1:
            # 单个图片直接复制
            src_path = os.path.join(img_dir, img_list[0])
            dst_path = os.path.join(label_dir, f"{id}.jpg")
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        else:
            try:
                # 多个图片需要拼接
                img_paths = [os.path.join(img_dir, img_name) for img_name in img_list]
                concat_img = concat_images(img_paths)
                base_name = os.path.splitext(img_list[0])[0]
                dst_path = os.path.join(label_dir, f"{id}_contact.jpg")
                concat_img.save(dst_path)
            except ValueError as e:
                print(f"处理图片列表时出错: {img_list}")
                print(f"错误信息: {str(e)}")
                continue

# 处理task2的图片
task2_dir = os.path.join(out_dir, 'task2')
os.makedirs(task2_dir, exist_ok=True)

for label, img_lists in tqdm(img_dict_task2.items(), desc='Processing Task2'):
    label_dir = os.path.join(task2_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    for i, (img_list, id) in enumerate(img_lists):
        if len(img_list) == 1:
            # 单个图片直接复制
            src_path = os.path.join(img_dir, img_list[0])
            dst_path = os.path.join(label_dir, f"{id}.jpg")
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        else:
            try:
                # 多个图片需要拼接
                img_paths = [os.path.join(img_dir, img_name) for img_name in img_list]
                concat_img = concat_images(img_paths)
                base_name = os.path.splitext(img_list[0])[0]
                dst_path = os.path.join(label_dir, f"{id}_contact.jpg")
                concat_img.save(dst_path)
            except ValueError as e:
                print(f"处理图片列表时出错: {img_list}")
                print(f"错误信息: {str(e)}")
                continue
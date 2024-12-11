import json
import os
from tqdm import tqdm
from paddleocr import PaddleOCR
import logging

# 设置日志级别忽略 DEBUG 信息
logging.getLogger("ppocr").setLevel(logging.WARNING)

json_file = {
    "train_task1": '/home/jch/data/www25/train/train_task1.json',
    "train_task2": '/home/jch/data/www25/train/train_task2.json',
    "test1_task1": '/home/jch/data/www25/test1/test1_task1.json',
    "test1_task2": '/home/jch/data/www25/test1/test1_task2.json'
}

output_file = {
    "train_task1": '/home/jch/projects/UserIntent/ocr_try/result/train_task1_ocr.json',
    "train_task2": '/home/jch/projects/UserIntent/ocr_try/result/train_task2_ocr.json',
    "test1_task1": '/home/jch/projects/UserIntent/ocr_try/result/test1_task1_ocr.json',
    "test1_task2": '/home/jch/projects/UserIntent/ocr_try/result/test1_task2_ocr.json'
}

ocr = PaddleOCR(use_gpu=True, use_angle_cls=True, lang='ch')

for key, value in json_file.items():
    with open(value, 'r', encoding='utf-8') as file:
        data = json.load(file)
    img_dir = '/home/jch/data/www25/train/images/' if 'train' in key else '/home/jch/data/www25/test1/images/'

    for item in tqdm(data):
        img_list = item['image']
        res_list = []
        for img in img_list:
            img_path = os.path.join(img_dir, img)
            try:
                result = ocr.ocr(img_path, cls=True)
                res_list.append(result)
            except Exception as e:
                logging.warning(f"Failed to process image {img_path}: {e}")
        item['ocr'] = res_list

    with open(output_file[key], 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

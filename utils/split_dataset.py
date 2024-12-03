import json

# 定义文件路径
file_path = '/home/pubw/datasets/www25/train/train.json'
# 生成新文件名
task1_name = file_path.replace('.json', '_task1.json')
task2_name = file_path.replace('.json', '_task2.json')

# 从原始文件加载数据
with open(file_path, 'r') as file:
    json_list = json.load(file)

# 初始化列表
task1_list = []
task2_list = []

# 根据条件对数据进行分类
for item in json_list:
    if 'Picture 1' in item.get('instruction', ''):
        task2_list.append(item)
    else:
        task1_list.append(item)

# 将任务列表保存为新的JSON文件
with open(task1_name, 'w', encoding='utf-8') as t1_file:
    json.dump(task1_list, t1_file, ensure_ascii=False, indent=4)

with open(task2_name, 'w', encoding='utf-8') as t2_file:
    json.dump(task2_list, t2_file,  ensure_ascii=False,indent=4)
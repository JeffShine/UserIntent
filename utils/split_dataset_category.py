import json

# 定义文件路径
file_path = '/home/pubw/datasets/www25/test1/test1.json'
# 生成新文件名
task1_name = file_path.replace('.json', '_task1_category.json')
task2_name = file_path.replace('.json', '_task2_category.json')

# 从原始文件加载数据
with open(file_path, 'r') as file:
    json_list = json.load(file)

# 初始化列表

task2_list=[]
task1_list=[]
# 根据条件对数据进行分类

for item in json_list:
    if 'Picture 1' in item.get('instruction', ''):
        task2_list.append(item)
    else:
        task1_list.append(item)

categroy_dict = {}
for item  in task1_list:
    if item['output'] not in categroy_dict:
        categroy_dict[item['output']]=[]
    categroy_dict[item['output']].append(item)

    # 将任务列表保存为新的JSON文件
with open(task1_name, 'w', encoding='utf-8') as t1_file:
    json.dump(categroy_dict, t1_file, ensure_ascii=False, indent=4)

categroy_dict = {}
for item  in task2_list:
    if item['output'] not in categroy_dict:
        categroy_dict[item['output']]=[]
    categroy_dict[item['output']].append(item)
with open(task2_name, 'w', encoding='utf-8') as t2_file:
    json.dump(categroy_dict, t2_file,  ensure_ascii=False,indent=4)
import json

# 定义两个JSON文件的路径
file_path1 = '/localnvme/application/sc_new/fmm/LLaMA-Factory-backup/data/mgsm8k_answer2_trans_task.json'
file_path2 = '/localnvme/application/sc_new/fmm/LLaMA-Factory-backup/data/mgsm8k_question_trans_task_new.json'

# 读取第一个JSON文件
with open(file_path1, 'r', encoding='utf-8') as file:
    data1 = json.load(file)

# 读取第二个JSON文件
with open(file_path2, 'r', encoding='utf-8') as file:
    data2 = json.load(file)

final_data = []
for data in data1:
    final_data.append(data)

for data in data2:
    final_data.append(data)

# 将合并后的数据写入新的JSON文件
with open('mgsm_trans_question_answer2.json', 'w', encoding='utf-8') as file:
    json.dump(final_data, file, ensure_ascii=False, indent=4)

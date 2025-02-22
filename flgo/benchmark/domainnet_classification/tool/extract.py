import os

# 定义要保留的类别
categories_to_keep = [
    'bird', 'feather', 'headphones', 'ice_cream', 'teapot',
    'tiger', 'whale', 'windmill', 'wine_glass', 'zebra'
]

# 定义输入和输出文件夹路径
input_folder = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet'
output_folder = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet_lite'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理每个文件
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line in infile:
                # 检查行中是否包含要保留的类别
                if any(category in line for category in categories_to_keep):
                    outfile.write(line)

print("过滤完成，结果已保存到", output_folder)
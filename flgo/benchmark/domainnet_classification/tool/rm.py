import os
import shutil

# 定义要保留的类别
categories_to_keep = [
    'bird', 'feather', 'headphones', 'ice_cream', 'teapot',
    'tiger', 'whale', 'windmill', 'wine_glass', 'zebra'
]

# 定义要处理的目录路径
# directory_path = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet/clipart'
# directory_path = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet/infograph'
# directory_path = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet/painting'
# directory_path = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet/quickdraw'
# directory_path = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet/real'
directory_path = '/mnt/d/dataset/easy fl/RAW_DATA/domainnet/sketch'

# 遍历目录中的所有子目录
for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)

    # 检查是否为目录且不在保留列表中
    if os.path.isdir(item_path) and item not in categories_to_keep:
        # 删除目录及其所有内容
        shutil.rmtree(item_path)
        print(f"已删除目录: {item_path}")

print("清理完成。")
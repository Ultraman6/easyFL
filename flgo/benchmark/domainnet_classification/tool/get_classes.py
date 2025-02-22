import os

def get_folder_names(path):
    """
    获取指定路径下的所有文件夹名称。

    Args:
        path (str): 要搜索的路径

    Returns:
        list: 包含文件夹名称的列表
    """
    if not os.path.isdir(path):
        raise ValueError(f"指定路径 {path} 不是有效的文件夹路径")

    # 获取文件夹名称列表
    folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return folder_names

def format_list_as_pystring(data, row_length=10):
    """
    将列表格式化为 Python 字符串列表，每行指定个数的元素。

    Args:
        data (list): 要格式化的列表
        row_length (int): 每行显示的元素数

    Returns:
        str: 格式化后的 Python 列表字符串
    """
    lines = []
    for i in range(0, len(data), row_length):
        line = ", ".join(f"'{item}'" for item in data[i:i + row_length])
        lines.append(f"    {line}")
    return "[\n" + ",\n".join(lines) + "\n]"


if __name__ == "__main__":
    # 设置目标路径
    target_path = '/Users/xyz/Documents/Datasets/RAW_DATA/domainnet/clipart'

    try:
        folder_list = get_folder_names(target_path)
        print("文件夹名称的 Python 列表格式:")
        py_string = format_list_as_pystring(folder_list, row_length=10)
        print(py_string)
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}")
import os
import glob

# def delete_impu_files():
#     """删除当前目录下所有以impu开头的文件"""
#     # 获取当前目录
#     current_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # 构建匹配模式
#     pattern = os.path.join(current_dir, "impu*")
    
#     # 查找所有匹配的文件
#     matching_files = glob.glob(pattern)
    
#     # 统计变量
#     total_files = len(matching_files)
#     deleted_files = 0
    
#     print(f"找到 {total_files} 个以impu开头的文件")
    
#     # 删除文件
#     for file_path in matching_files:
#         try:
#             os.remove(file_path)
#             print(f"已删除: {os.path.basename(file_path)}")
#             deleted_files += 1
#         except Exception as e:
#             print(f"删除 {os.path.basename(file_path)} 时出错: {e}")
    
#     print(f"操作完成: 成功删除 {deleted_files}/{total_files} 个文件")

# if __name__ == "__main__":
#     delete_impu_files()、

import numpy as np
import pandas as pd
import os
from baseline import *
import warnings
warnings.filterwarnings('ignore')

def apply_basic_imputation_methods(df):
    """对整个DataFrame应用基本填充方法"""
    results = {}
    
    # 转换为numpy数组进行处理
    data_array = df.values
    
    # 基础填充方法
    basic_methods = {
        # 'zero': zero_impu,
        # 'mean': mean_impu,
        # 'median': median_impu,
        # 'mode': mode_impu,
        # 'random': random_impu,
        # 'knn': knn_impu,
        # 'ffill': ffill_impu,
        # 'bfill': bfill_impu,
        'mice': mice_impu,
    }
    
    print(f"原始数据形状: {data_array.shape}")
    print(f"总缺失值数量: {np.isnan(data_array).sum()}")
    print(f"缺失值比例: {np.isnan(data_array).sum() / data_array.size:.2%}")
    
    # 应用基础填充方法
    for method_name, method_func in basic_methods.items():
        try:
            print(f"\n正在执行 {method_name} 填充...")
            imputed_data = method_func(data_array.copy())
            
            # 验证结果
            if imputed_data is None:
                print(f"⚠️ {method_name} 返回None，跳过")
                continue
            
            if imputed_data.shape != data_array.shape:
                print(f"⚠️ {method_name} 返回形状不匹配: {imputed_data.shape} vs {data_array.shape}")
                continue
            
            # 转换回DataFrame
            imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
            results[method_name] = imputed_df
            
            remaining_missing = np.isnan(imputed_data).sum()
            print(f"✓ {method_name} 填充完成，剩余缺失值: {remaining_missing}")
            
        except Exception as e:
            print(f"✗ {method_name} 填充失败: {e}")
            continue
    
    return results

def apply_advanced_imputation_methods(df):
    """对整个DataFrame应用高级填充方法"""
    results = {}
    
    # 转换为numpy数组进行处理
    data_array = df.values
    
    # 高级填充方法
    advanced_methods = {
        # 'miracle': miracle_impu,
        # 'saits': saits_impu,
        # 'timemixerpp': timemixerpp_impu,
        # 'tefn': tefn_impu,
        # 'timesnet': timesnet_impu,
        'tsde': tsde_impu,
        'grin': grin_impu,
    }
    
    # 应用高级填充方法
    for method_name, method_func in advanced_methods.items():
        try:
            print(f"\n正在执行 {method_name} 填充...")
            imputed_data = method_func(data_array.copy())
            
            # 验证结果
            if imputed_data is None:
                print(f"⚠️ {method_name} 返回None，跳过")
                continue
            
            if imputed_data.shape != data_array.shape:
                print(f"⚠️ {method_name} 返回形状不匹配: {imputed_data.shape} vs {data_array.shape}")
                continue
            
            # 转换回DataFrame
            imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
            results[method_name] = imputed_df
            
            remaining_missing = np.isnan(imputed_data).sum()
            print(f"✓ {method_name} 填充完成，剩余缺失值: {remaining_missing}")
            
        except Exception as e:
            print(f"✗ {method_name} 填充失败: {e}")
            continue
    
    return results

def save_imputation_results(original_df, basic_results, advanced_results, output_dir="./", file_prefix="imputed"):
    """保存所有填充结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存原始数据（未填充版本）
    original_path = os.path.join(output_dir, f"{file_prefix}_original.csv")
    original_df.to_csv(original_path, index=False)
    print(f"✓ 原始数据已保存到: {original_path}")
    
    saved_files = [original_path]
    
    # 保存基础方法填充结果
    for method_name, imputed_df in basic_results.items():
        output_path = os.path.join(output_dir, f"{file_prefix}_basic_{method_name}.csv")
        imputed_df.to_csv(output_path, index=False)
        print(f"✓ {method_name} 填充结果已保存到: {output_path}")
        saved_files.append(output_path)
    
    # 保存高级方法填充结果
    for method_name, imputed_df in advanced_results.items():
        output_path = os.path.join(output_dir, f"{file_prefix}_advanced_{method_name}.csv")
        imputed_df.to_csv(output_path, index=False)
        print(f"✓ {method_name} 填充结果已保存到: {output_path}")
        saved_files.append(output_path)
    
    return saved_files

def generate_summary_report(original_df, basic_results, advanced_results, output_dir="./", file_prefix="imputed"):
    """生成填充效果总结报告"""
    summary_data = []
    
    # 原始数据统计
    original_missing = original_df.isna().sum().sum()
    total_cells = original_df.size
    original_missing_ratio = original_missing / total_cells
    
    summary_data.append({
        'Method': 'Original',
        'Type': 'None',
        'Missing_Count': original_missing,
        'Missing_Ratio': f"{original_missing_ratio:.2%}",
        'Total_Cells': total_cells,
        'Success': True
    })
    
    # 基础方法统计
    for method_name, imputed_df in basic_results.items():
        remaining_missing = imputed_df.isna().sum().sum()
        missing_ratio = remaining_missing / total_cells
        
        summary_data.append({
            'Method': method_name,
            'Type': 'Basic',
            'Missing_Count': remaining_missing,
            'Missing_Ratio': f"{missing_ratio:.2%}",
            'Total_Cells': total_cells,
            'Success': True
        })
    
    # 高级方法统计
    for method_name, imputed_df in advanced_results.items():
        remaining_missing = imputed_df.isna().sum().sum()
        missing_ratio = remaining_missing / total_cells
        
        summary_data.append({
            'Method': method_name,
            'Type': 'Advanced',
            'Missing_Count': remaining_missing,
            'Missing_Ratio': f"{missing_ratio:.2%}",
            'Total_Cells': total_cells,
            'Success': True
        })
    
    # 保存总结报告
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f"{file_prefix}_summary_report.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ 填充效果总结报告已保存到: {summary_path}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"填充效果总结")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"{'='*60}")
    
    return summary_path

def main(file_path, output_dir="./"):
    """主函数"""
    print(f"开始处理文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    # 读取文件
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print("不支持的文件格式")
            return
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 获取文件基名作为前缀
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    
    print(f"数据形状: {df.shape}")
    print(f"总缺失值数量: {df.isna().sum().sum()}")
    print(f"缺失值比例: {df.isna().sum().sum() / df.size:.2%}")
    
    # 应用基础填充方法
    print(f"\n{'='*50}")
    print("开始应用基础填充方法")
    print(f"{'='*50}")
    basic_results = apply_basic_imputation_methods(df)
    
    # 应用高级填充方法
    print(f"\n{'='*50}")
    print("开始应用高级填充方法")
    print(f"{'='*50}")
    advanced_results = apply_advanced_imputation_methods(df)
    
    # 保存所有结果
    print(f"\n{'='*50}")
    print("保存填充结果")
    print(f"{'='*50}")
    saved_files = save_imputation_results(df, basic_results, advanced_results, output_dir, file_basename)
    
    # 生成总结报告
    summary_path = generate_summary_report(df, basic_results, advanced_results, output_dir, file_basename)
    
    # 最终总结
    print(f"\n🎉 处理完成！")
    print(f"共生成 {len(saved_files) + 1} 个文件：")
    for file in saved_files:
        print(f"  - {file}")
    print(f"  - {summary_path}")

if __name__ == "__main__":
    import sys
    
    # 默认参数
    default_file = "/data/zhangxian/newCode2/data/III/200001.csv"
    default_output = "./"
    
    # 从命令行获取参数
    if len(sys.argv) >= 2:
        target_file = sys.argv[1]
    else:
        target_file = default_file
    
    if len(sys.argv) >= 3:
        output_directory = sys.argv[2]
    else:
        output_directory = default_output
    
    print(f"目标文件: {target_file}")
    print(f"输出目录: {output_directory}")
    
    main(target_file, output_directory)
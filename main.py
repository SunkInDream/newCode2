from models_impute import *
from models_downstream import *
from baseline import *
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
model_params = {
        'num_levels': 10,
        'kernel_size': 6,
        'dilation_c': 2
    }
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_arr = Prepare_data('./data/')
    cg = causal_discovery(data_arr, 20)
    # res = parallel_impute_folder(cg, './data', model_params, epochs=150, lr=0.02)
    # parellel_mse_compare(res, cg)
    
    
    data_arr1, label_arr1 = Prepare_data('./data/', './static_tag.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
     # 评估所有插补方法
    results = evaluate_imputation_methods(data_arr1, label_arr1, k=4, epochs=100, lr=0.02)

    table = []

    for method, metrics in results.items():
        row = {
            'Method': method,
            'Accuracy (mean ± std)': f"{metrics['Accuracy'][0]:.2%} ± {metrics['Accuracy'][1]:.2%}",
            'Precision (mean ± std)': f"{metrics['Precision'][0]:.2%} ± {metrics['Precision'][1]:.2%}",
            'Recall (mean ± std)': f"{metrics['Recall'][0]:.2%} ± {metrics['Recall'][1]:.2%}",
            'F1 Score (mean ± std)': f"{metrics['F1'][0]:.2%} ± {metrics['F1'][1]:.2%}",
            'AUROC (mean ± std)': f"{metrics['AUROC'][0]:.4f} ± {metrics['AUROC'][1]:.4f}",
        }
        table.append(row)

    df_results = pd.DataFrame(table)
    
    print(df_results)

    # 或保存为 CSV 文件
    df_results.to_csv('imputation_comparison_results.csv', index=False)


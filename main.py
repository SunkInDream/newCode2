from models_impute import *
from models_downstream import *
from baseline import *
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
model_params = {
        'num_levels': 8,
        'kernel_size': 6,
        'dilation_c': 2
    }
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_arr = prepare_data('./data/')
    cg = causal_discovery(data_arr, 20)
    # res = parallel_impute_folder(cg, './data', model_params, epochs=100, lr=0.02)
    # parellel_mse_compare(res, cg)
    
    
    data_arr1, label_arr1 = prepare_data('./data/', './static_tag.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
    data_arr2 = parallel_impute_folder(cg, './data', model_params, epochs=150, lr=0.02)
    accs = train_and_evaluate(data_arr2, label_arr1, k=3, epochs=150, lr=0.02)
    data_arr3 = [zero_impu(matrix) for matrix in data_arr1]
    accs2 = train_and_evaluate(data_arr3, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr4 = [median_impu(matrix) for matrix in data_arr1]
    accs3 = train_and_evaluate(data_arr4, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr5 = [mode_impu(matrix) for matrix in data_arr1]
    accs4 = train_and_evaluate(data_arr5, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr6 = [random_impu(matrix) for matrix in data_arr1]
    accs5 = train_and_evaluate(data_arr6, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr7 = [knn_impu(matrix) for matrix in data_arr1]
    accs6 = train_and_evaluate(data_arr7, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr8 = [mean_impu(matrix) for matrix in data_arr1]
    accs7 = train_and_evaluate(data_arr8, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr9 = [ffill_impu(matrix) for matrix in data_arr1]
    accs8 = train_and_evaluate(data_arr9, label_arr1, k=3, epochs=100, lr=0.02)
    data_arr10 = [bfill_impu(matrix) for matrix in data_arr1]
    accs9 = train_and_evaluate(data_arr10, label_arr1, k=3, epochs=100, lr=0.02)    
    
    results = {
    'Causal-Impute': accs,
    'Zero-Impute': accs2,
    'Median-Impute': accs3,
    'Mode-Impute': accs4,
    'Random-Impute': accs5,
    'KNN-Impute': accs6,
    'Mean-Impute': accs7,
    'FFill-Impute': accs8,
    'BFill-Impute': accs9,
    }

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


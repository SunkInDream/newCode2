from models_impute import *
from models_downstream import *
from baseline import *
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_arr, label_arr = prepare_data('./data/', './static_tag.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
    print(data_arr[0].dtype)  # 确认数据类型
    cg = causal_discovery(data_arr, 3)#
    print(cg)
    dat = data_arr[0]
    print(dat)
    input_dir = "./data"
    model_params = {
        'num_levels': 6,
        'kernel_size': 6,
        'dilation_c': 4
    }

    res = parallel_impute_folder(cg, input_dir, model_params, epochs=100, lr=0.02)
    # …前面已有 res = parallel_impute_folder(…) …
    print(f"共 {len(res)} 个矩阵，开始批量 MSE 评估…")

    mse_dicts = parallel_mse_evaluate(res, causal_matrix=cg)

    valid_mse_dicts = [d for d in mse_dicts if d is not None]
    
    if not valid_mse_dicts:
        print("错误: 所有MSE评估任务均失败!")
    else:
        print(f"成功完成 {len(valid_mse_dicts)}/{len(mse_dicts)} 个MSE评估")
        
        # 计算平均MSE
        avg_mse = {}
        for method in valid_mse_dicts[0]:
            vals = [d[method] for d in valid_mse_dicts if d is not None]
            if vals:
                avg_mse[method] = sum(vals) / len(vals)
            else:
                avg_mse[method] = float('nan')
        
        print("各方法平均 MSE:")
        for method, v in sorted(avg_mse.items()):
            print(f"{method:12s}: {v:.6f}")

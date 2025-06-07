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
    cg = causal_discovery(data_arr, 4)
    
    # res = parallel_impute_folder(cg, './data', model_params, epochs=150, lr=0.02)
    # parellel_mse_compare(res, cg)
    
    data_arr1, label_arr1 = Prepare_data('./data/', './static_tag.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
     # 评估所有插补方法
    results = evaluate_downstream(data_arr1, label_arr1, k=4, epochs=100, lr=0.02)

    


from models_impute import *
from models_downstream import *
from baseline import *
model_params = {
        'num_levels': 6,
        'kernel_size': 6,
        'dilation_c': 4
    }
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_arr = prepare_data('./data/')
    cg = causal_discovery(data_arr, 5)
    # res = parallel_impute_folder(cg, './data', model_params, epochs=100, lr=0.02)
    # parellel_mse_compare(res, cg)
    
    
    data_arr2, label_arr2 = prepare_data('./data/', './static_tag.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
    data_arr2 = parallel_impute_folder(cg, './data', model_params, epochs=100, lr=0.02)
    #data_arr2 = zero_impu(data_arr2)    
    accs = train_and_evaluate(data_arr2, label_arr2, k=5, epochs=100, lr=0.02)


from models_impute import *
from models_downstream import *
from baseline import *
# torch.set_printoptions(threshold=float('inf'))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
model_params = {
        'num_levels': 8,
        'kernel_size': 8,
        'dilation_c': 6
    }

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    data_arr = Prepare_data('./data/lorenz')
    cg = causal_discovery(data_arr, 20, isStandard=True, standard_cg='./causality_matrices/lorenz_causality_matrix.csv')
    
    # res = parallel_impute('./data/mimic-iii/III', cg, model_params, epochs=150, lr=0.02)
    parallel_mse_evaluate(data_arr, cg)
    
    # data_arr1, label_arr1 = Prepare_data('./data_imputed/my_model/mimic-iii', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
    # results = evaluate_downstream(data_arr1, label_arr1, k=4, epochs=100, lr=0.02)

    


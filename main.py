from models_impute import *
from models_downstream import *
from models_TCDF import *
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
    data_arr = Prepare_data('./data/air')
    cg = causal_discovery(data_arr, 20, isStandard=False, standard_cg='./causality_matrices/finance_causality_matrix.csv')

    tar = pd.read_csv('./2013-03-07_dig_missing.csv')
    res,_,_ = impute(tar,cg,model_params,gpu_id=0)
    pd.DataFrame(res).to_csv('./air_dataset_0_timeseries_dig_missing_basic_scit.csv', index=False)

    # pd.DataFrame(cg).to_csv('./causality_matrices/my_lorenz_causality_matrix.csv', index=False, header=False)
    # res = evaluate_causal_discovery_from_file('./causality_matrices/my_lorenz_causality_matrix.csv', './causality_matrices/lorenz_causality_matrix.csv')
    # print(res)

    # res = parallel_impute('./data/mimic-iii', cg, model_params, epochs=100, lr=0.02, output_dir='./data_imputed/my_model/mimic-iii')
    # parallel_mse_evaluate(data_arr, cg)

    # data_arr1, label_arr1 = Prepare_data('./data_imputed/my_model/mimic-iii', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', 'DIEINHOSPITAL')
    # results = evaluate_downstream(data_arr1, label_arr1, k=4, epochs=100, lr=0.02)

    


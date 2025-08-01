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
    SEED = 33
    set_seed(SEED)
    met = 'lorenz'
    missing = 'mar'
    tag = 'FirstICU24_AKI_ALL'

    # data_arr = Prepare_data(f'./data/{met}')
    # cg = causal_discovery(data_arr, 20, isStandard=True, standard_cg=f'./causality_matrices/{met}_causality_matrix.csv',met=met)

    # tar = pd.read_csv('./2013-03-07_dig_missing.csv')
    # res,_,_ = impute(tar,cg,model_params,gpu_id=0)
    # pd.DataFrame(res).to_csv('./air_dataset_0_timeseries_dig_missing_basic_scit.csv', index=False)

    # pd.DataFrame(cg).to_csv('./causality_matrices/my_lorenz_causality_matrix.csv', index=False, header=False)
    # res = evaluate_causal_discovery_from_file('./causality_matrices/my_lorenz_causality_matrix.csv', './causality_matrices/lorenz_causality_matrix.csv')
    # print(res)

    # res = parallel_impute('./data/downstreamIII', cg, model_params, epochs=100, lr=0.02, output_dir='./data_imputed/my_model/III', skip_existing=True)
    # parallel_mse_evaluate(data_arr, cg, met=met, missing=missing, seed=SEED, ablation=0)

    data_arr1, label_arr1 = Prepare_data('./data/downstreamIII', './AAAI_3_4_labels.csv', 'ICUSTAY_ID', tag)
    results = evaluate_downstream(data_arr1, label_arr1, k=4, epochs=100, lr=0.02, seed=SEED, tag=tag)

    


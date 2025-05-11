import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import queue
from sklearn.cluster import KMeans
from models_CAUSAL import *
from models_TCDF import ADDSTCN
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from models_impute import process_single_matrix
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
def compute_single_causal(args):
    mat, params, gpu = args
    return compute_causal_matrix(mat, params, gpu)[0]
def eval_worker(gpu_id, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    print(f"[PID {os.getpid()}] 启动进程，绑定 GPU {gpu_id} → 设备 {device}")

    while True:
        try:
            args = task_queue.get(timeout=5)
        except queue.Empty:
            break
        if args is None:
            break

        idx, mat, mask_eva, train_mask, test_idx, total_causal_matrix, model_params, epochs, lr, knn_k = args
        print(f"[GPU {gpu_id}] 正在处理矩阵 {idx}")

        from models_impute import process_single_matrix
        _, ours = process_single_matrix((idx, mat, train_mask, total_causal_matrix, model_params, epochs, lr, device))

        y = mat.flatten()[test_idx]
        res = {'idx': idx, 'ours': ((ours.flatten()[test_idx] - y) ** 2).mean()}
        missing = train_mask == 0
        f0 = mat.copy(); f0[missing] = 0
        res['zero'] = ((f0.flatten()[test_idx] - y) ** 2).mean()
        med = np.median(mat[train_mask == 1])
        fm = mat.copy(); fm[missing] = med
        res['median'] = ((fm.flatten()[test_idx] - y) ** 2).mean()
        mu = np.mean(mat[train_mask == 1])
        fmu = mat.copy(); fmu[missing] = mu
        res['mean'] = ((fmu.flatten()[test_idx] - y) ** 2).mean()
        df = pd.DataFrame(mat)
        dfm = df.mask(~train_mask.astype(bool))
        res['ffill'] = ((dfm.ffill().fillna(0).values.flatten()[test_idx] - y) ** 2).mean()
        res['bfill'] = ((dfm.bfill().fillna(0).values.flatten()[test_idx] - y) ** 2).mean()
        knn_imp = KNNImputer(n_neighbors=knn_k)
        res['knn'] = ((knn_imp.fit_transform(dfm).flatten()[test_idx] - y) ** 2).mean()
        mice_imp = IterativeImputer(max_iter=10, random_state=0)
        res['mice'] = ((mice_imp.fit_transform(dfm).flatten()[test_idx] - y) ** 2).mean()

        result_queue.put(res)



class MyDataset(Dataset):
    def __init__(self, file_paths): 
        self.file_paths = file_paths  
        self.original_data = []        # 含nan的原始数据 (list of ndarray)
        self.mask_data = []            # 掩码矩阵，1表示有值，0表示缺失 (list of ndarray)
        self.initial_filled = []       # 初次填补的结果 (list of ndarray)
        self.final_filled = []         # 最终填补结果 (list of ndarray, 默认为None)
        self.center_repre = []       # 中心表示 (list of ndarray)
        self.labels = []          # 标签 (list)
        self.total_causal_matrix = None  # 总因果矩阵 (ndarray, 默认为None)
        files = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
        for file in files:
            df = pd.read_csv(file)
            data = df.values.astype(np.float32)
            self.original_data.append(data)
            first_prepro_data =  FirstProcess(file)
            mask = (~first_prepro_data.isna()).values.astype(np.float32)
            self.mask_data.append(mask)
            second_prepro_data = SecondProcess(first_prepro_data)
            second_prepro_data = second_prepro_data.values.astype(np.float32)
            self.initial_filled.append(second_prepro_data)
            self.final_filled.append(None)
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return {
            'original': self.original_data[idx],
            'mask': self.mask_data[idx],
            'initial_filled': self.initial_filled[idx],
            'final_filled':self.final_filled[idx]
            #'label': self.labels[idx],
            #'total_causal_matrix': torch.tensor(self.total_causal_matrix) if self.total_causal_matrix is not None else None
        }
    def agr(self, n_cluster, params=None):
        # 仅执行聚类，返回中心表示
        data = np.array([np.nanmean(x, axis=0) for x in self.initial_filled])
        data = np.nan_to_num(data, nan=0.0)
        km = KMeans(n_clusters=n_cluster, n_init=10, random_state=0)
        labels = km.fit_predict(data)

        self.labels = labels  # 保存标签
        self.center_repre = []
        
        for k in range(n_cluster):
            idxs = np.where(labels == k)[0]
            if len(idxs) == 0:  # 处理空聚类
                continue
            cluster_data = data[idxs]
            dists = np.linalg.norm(cluster_data - km.cluster_centers_[k], axis=1)
            best_idx = idxs[np.argmin(dists)]
            self.center_repre.append(self.initial_filled[best_idx])
            
        return self.center_repre  # 返回中心表示，不进行因果发现  
    def evaluate(self, k_folds=5, point_prob=0.1, block_prob=0.2,
             block_min=1, block_max=5, model_params={},
             knn_k=5, epochs=10, lr=0.001):

        def gen_mask(mat):
            m = (np.random.rand(*mat.shape) > point_prob).astype(np.float32)
            if np.random.rand() < block_prob:
                h = np.random.randint(block_min, block_max + 1)
                w = np.random.randint(block_min, block_max + 1)
                i = np.random.randint(0, mat.shape[0] - h + 1)
                j = np.random.randint(0, mat.shape[1] - w + 1)
                m[i:i + h, j:j + w] = 0
            return m

        gpus = list(range(torch.cuda.device_count()))
        print(f"可用GPU: {gpus}")

        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        for idx, mat in enumerate(self.final_filled):
            mask_eva = gen_mask(mat)
            avail = np.where(mask_eva.flatten() == 1)[0]
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
            for _, test_split in kf.split(avail):
                test_idx = avail[test_split]
                train_mask = mask_eva.flatten()
                train_mask[test_idx] = 0
                train_mask = train_mask.reshape(mat.shape)
                task_queue.put((idx, mat, mask_eva, train_mask, test_idx,
                                self.total_causal_matrix, model_params, epochs, lr, knn_k))

        for _ in gpus:  # 哨兵 None 终止
            task_queue.put(None)

        workers = []
        for gpu_id in gpus:
            p = multiprocessing.Process(target=eval_worker, args=(gpu_id, task_queue, result_queue))
            p.start()
            workers.append(p)

        results = []
        expected = task_queue.qsize() - len(gpus)
        for _ in range(expected):
            results.append(result_queue.get())

        for p in workers:
            p.join()

        df = pd.DataFrame(results).groupby('idx').mean()
        print(df.T)



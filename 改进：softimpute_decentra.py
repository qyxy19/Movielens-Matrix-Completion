import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import time
import warnings
import os

warnings.filterwarnings('ignore')

# 1. 数据加载类
class MovieLensData:
    def __init__(self, ratings_file):
        self.ratings_file = ratings_file
    def load_ratings(self):
        try:
            #  UserID::MovieID::Rating::Timestamp
            if self.ratings_file and self.ratings_file.endswith('.dat'):
                df = pd.read_csv(self.ratings_file, sep='::', engine='python',
                                 header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
                print(f"成功从.dat文件加载{len(df)}条评分")
            else:
                # 尝试其他格式
                try:
                    df = pd.read_csv(self.ratings_file)
                    print(f"成功从.csv文件加载{len(df)}条评分")
                except:
                    raise Exception("无法加载数据文件")
            # 创建映射 - 从原始ID到0-based索引
            self.user_ids = np.sort(df['user_id'].unique())
            self.movie_ids = np.sort(df['movie_id'].unique())

            self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
            self.movie_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

            self.n_users = len(self.user_ids)
            self.n_movies = len(self.movie_ids)

            # 创建稀疏矩阵
            rows = df['user_id'].map(self.user_to_idx).to_numpy()
            cols = df['movie_id'].map(self.movie_to_idx).to_numpy()
            vals = df['rating'].to_numpy().astype(np.float32)

            # 创建稀疏矩阵
            self.R = sp.coo_matrix((vals, (rows, cols)),
                                   shape=(self.n_users, self.n_movies)).tocsr()
            # 存储原始数据用于交叉验证
            self.ratings_list = list(zip(rows, cols, vals))
            density = len(self.ratings_list) / (self.n_users * self.n_movies) * 100
            print(
                f"数据统计: 用户数={self.n_users}, 电影数={self.n_movies}, 评分总数={len(self.ratings_list)}, 密度={density:.4f}%")

            return self.R

        except Exception as e:
            print(f"加载数据失败: {e}")
            print("使用测试数据...")
            return self.create_test_data()

    def create_test_data(self):
        """创建测试数据"""
        print("使用测试数据...")
        self.n_users = 100
        self.n_movies = 200

        # 创建低秩矩阵
        rank = 5
        U = np.random.randn(self.n_users, rank) * 0.5
        V = np.random.randn(self.n_movies, rank) * 0.5
        M = U @ V.T

        # 添加噪声并缩放到1-5范围
        M = np.clip(M + np.random.randn(*M.shape) * 0.1, 1, 5)

        # 创建稀疏观测
        mask = np.random.rand(self.n_users, self.n_movies) < 0.1
        rows, cols = np.where(mask)
        vals = M[rows, cols]

        self.R = sp.coo_matrix((vals, (rows, cols)),
                               shape=(self.n_users, self.n_movies)).tocsr()

        # 存储原始数据
        self.ratings_list = list(zip(rows, cols, vals))
        self.user_to_idx = {i: i for i in range(self.n_users)}
        self.movie_to_idx = {i: i for i in range(self.n_movies)}

        print(f"测试数据: 用户数={self.n_users}, 电影数={self.n_movies}, 评分数={len(self.ratings_list)}")

        return self.R

    def create_cross_validation_splits(self, n_folds=5, seed=42):

        np.random.seed(seed)
        n_ratings = len(self.ratings_list)

        # 随机打乱索引
        indices = np.random.permutation(n_ratings)

        fold_size = n_ratings // n_folds
        remainder = n_ratings % n_folds

        splits = []
        start = 0

        for i in range(n_folds):
            test_size = fold_size + (1 if i < remainder else 0)
            test_indices = indices[start:start + test_size]
            train_indices = np.concatenate([indices[:start], indices[start + test_size:]])
            splits.append((train_indices, test_indices))
            start += test_size

        print(f"创建了{n_folds}折交叉验证分割")
        return splits


# 2. Soft-Impute算法
class EfficientSoftImpute:

    def __init__(self, rank=50, lambda_reg=80, max_iter=15, tol=1e-4,
                 randomized=True, n_iter_svd=7, verbose=True):
        """
        参数初始化
        """
        self.rank = int(rank)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.randomized = bool(randomized)
        self.n_iter_svd = int(n_iter_svd)
        self.verbose = bool(verbose)

        # 存储中心化统计量
        self.mu = None
        self.b_u = None
        self.b_i = None

        # 低秩因子
        self.U_k = None
        self.s_k = None
        self.Vt_k = None

        # 训练历史
        self.iter_history = []

    def compute_centering_stats(self, R_train):
        rows, cols = R_train.nonzero()
        ratings = R_train.data.astype(np.float32)
        mu = float(ratings.mean())

        n_users, n_items = R_train.shape

        # 用户偏置
        sum_u = np.bincount(rows, weights=ratings, minlength=n_users).astype(np.float32)
        counts_u = np.bincount(rows, minlength=n_users).astype(np.int32)
        b_u = np.zeros(n_users, dtype=np.float32)
        mask_u = counts_u > 0
        b_u[mask_u] = sum_u[mask_u] / counts_u[mask_u] - mu

        # 物品偏置
        sum_i = np.bincount(cols, weights=ratings, minlength=n_items).astype(np.float32)
        counts_i = np.bincount(cols, minlength=n_items).astype(np.int32)
        b_i = np.zeros(n_items, dtype=np.float32)
        mask_i = counts_i > 0
        b_i[mask_i] = sum_i[mask_i] / counts_i[mask_i] - mu

        return mu, b_u, b_i

    def _truncated_svd(self, X, k):
        if self.randomized:
            # 随机SVD，效率更高
            U, s, Vt = randomized_svd(
                X, n_components=k, n_iter=self.n_iter_svd, random_state=42
            )
            return U.astype(np.float32), s.astype(np.float32), Vt.astype(np.float32)
        else:
            # 标准SVD
            X64 = X.astype(np.float64, copy=False)
            U64, s64, Vt64 = svds(X64, k=k)
            # svds返回的奇异值从小到大，需要翻转
            s_order = s64[::-1]
            U_order = U64[:, ::-1]
            Vt_order = Vt64[::-1, :]
            return (U_order.astype(np.float32),
                    s_order.astype(np.float32),
                    Vt_order.astype(np.float32))

    def fit(self, R_train):
        # 1) 计算中心化统计量
        if self.verbose:
            print("计算中心化统计量...")

        self.mu, self.b_u, self.b_i = self.compute_centering_stats(R_train)

        m, n = R_train.shape
        k = min(self.rank, min(m, n) - 1)

        if k <= 0:
            self.U_k = None
            self.s_k = None
            self.Vt_k = None
            self.iter_history = []
            return self

        # 2) 提取稀疏观测并中心化
        rows, cols = R_train.nonzero()
        vals = R_train.data.astype(np.float32)

        # 中心化：r - mu - b_u - b_i
        centered_vals = vals - (self.mu + self.b_u[rows] + self.b_i[cols])

        if self.verbose:
            print(f"矩阵大小: {m}×{n}，观测数: {len(centered_vals)}")
            print(f"开始Soft-Impute迭代 (rank={k}, lambda={self.lambda_reg})...")

        # 3) 初始化填充矩阵 - 只在观测位置有值
        # 对于大数据集，我们只构建必要的数据结构
        X = np.zeros((m, n), dtype=np.float32)
        X[rows, cols] = centered_vals

        # 4) Soft-Impute迭代
        prev_frob = float('inf')
        self.iter_history = []
        fit_start_time = time.time()

        for it in range(self.max_iter):
            # a) SVD分解
            U, s, Vt = self._truncated_svd(X, k=k)

            # b) 软阈值
            s_shrunk = np.maximum(s - self.lambda_reg, 0.0)

            # c) 统计非零奇异值
            nz = s_shrunk > 0

            # d) 重构矩阵
            if nz.sum() == 0:
                X_recon = np.zeros_like(X)
            else:
                U_k = U[:, nz]
                s_k = s_shrunk[nz]
                Vt_k = Vt[nz, :]

                # 高效重构：X_recon = U_k * diag(s_k) * Vt_k
                V_k = (Vt_k.T * s_k.reshape(1, -1))
                X_recon = U_k @ V_k.T

            # e) 计算观测位置的重构误差
            preds_obs_recon = X_recon[rows, cols]
            obs_mse_recon = float(np.mean((centered_vals - preds_obs_recon) ** 2))

            # f) 在观测位置覆盖原始值
            X_new = X_recon.copy()
            X_new[rows, cols] = centered_vals

            # g) 收敛判定
            frob_diff = np.linalg.norm(X_new - X, ord='fro')

            elapsed = time.time() - fit_start_time
            self.iter_history.append((elapsed, obs_mse_recon))

            if self.verbose:
                topk = min(5, len(s))
                print(f"[{it + 1}/{self.max_iter}] "
                      f"MSE={obs_mse_recon:.6f}, "
                      f"frob_diff={frob_diff:.6f}, "
                      f"保留奇异值={nz.sum()}/{len(s)}, "
                      f"时间={elapsed:.1f}s")

            # 更新X
            X = X_new

            if frob_diff < self.tol:
                if self.verbose:
                    print(f"收敛于第{it + 1}次迭代")
                break

        # 5) 保存最终低秩因子
        if nz.sum() > 0:
            self.U_k = U_k
            self.s_k = s_k
            self.Vt_k = Vt_k
        else:
            self.U_k = None
            self.s_k = None
            self.Vt_k = None

        total_time = time.time() - fit_start_time
        if self.verbose:
            print(f"训练完成，总时间: {total_time:.1f}秒")
            if self.U_k is not None:
                print(f"最终秩: {self.U_k.shape[1]}")

        return self

    def predict(self, user_idx, item_idx):
        """
        预测评分 - 支持单个值或数组
        """
        # 处理单个值的情况
        if np.isscalar(user_idx):
            user_idx = np.array([user_idx])
        if np.isscalar(item_idx):
            item_idx = np.array([item_idx])

        # 确保是数组
        uu = np.atleast_1d(np.array(user_idx, dtype=np.int64))
        vv = np.atleast_1d(np.array(item_idx, dtype=np.int64))

        # 检查长度是否匹配
        if len(uu) != len(vv):
            if len(uu) == 1:
                uu = np.full(len(vv), uu[0])
            elif len(vv) == 1:
                vv = np.full(len(uu), vv[0])
            else:
                raise ValueError("user_idx和item_idx的长度必须相同或其中之一长度为1")

        if self.U_k is None or self.s_k is None or self.Vt_k is None:
            # 全0 => 预测=中心化+偏置=mu+bias
            preds = self.mu + self.b_u[uu] + self.b_i[vv]
        else:
            # 计算中心化预测
            n = len(uu)
            preds_center = np.empty(n, dtype=np.float32)

            # 批量处理以提高效率
            batch_size = 10000
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                up = uu[start:end]
                vp = vv[start:end]

                U_part = self.U_k[up, :]
                V_part = self.Vt_k[:, vp].T
                preds_center[start:end] = np.sum(U_part * (self.s_k * V_part), axis=1)

            # 恢复偏置
            preds = preds_center + self.mu + self.b_u[uu] + self.b_i[vv]

        # 限制评分范围
        preds = np.clip(preds, 1.0, 5.0)

        # 如果输入是单个值，返回单个值
        if np.isscalar(user_idx) and np.isscalar(item_idx):
            return float(preds[0])

        return preds.flatten()


# 3. 实验和评估类
class SoftImputeExperiment:
    """Soft-Impute实验框架"""
    def __init__(self, data_loader, n_folds=5, history_dir="history"):
        self.data_loader = data_loader
        self.n_folds = n_folds
        self.history_dir = history_dir
        self.results = {}
        os.makedirs(self.history_dir, exist_ok=True)

    def create_train_test_matrices(self, train_indices, test_indices):
        """创建训练和测试矩阵"""
        # 提取训练数据
        train_data = [self.data_loader.ratings_list[i] for i in train_indices]
        rows_train = [u for u, _, _ in train_data]
        cols_train = [m for _, m, _ in train_data]
        vals_train = [r for _, _, r in train_data]

        # 创建训练矩阵
        R_train = sp.coo_matrix((vals_train, (rows_train, cols_train)),
                                shape=(self.data_loader.n_users,
                                       self.data_loader.n_movies)).tocsr()

        # 提取测试数据
        test_data = [self.data_loader.ratings_list[i] for i in test_indices]

        return R_train, test_data

    def _save_history(self, model_name, fold, iter_history, final_metric, total_fit_time):
        """保存历史记录"""
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(self.history_dir, f"{safe_name}_fold{fold}_history.npz")

        if iter_history:
            arr = np.array(iter_history, dtype=np.float64)
        else:
            arr = np.zeros((0, 2), dtype=np.float64)

        np.savez_compressed(
            path,
            iter_history=arr,
            final_metric=float(final_metric),
            total_fit_time=float(total_fit_time)
        )
        return path

    def safe_compute_rmse(self, test_preds, test_actuals, batch_size=100000):
        """安全计算RMSE，避免内存问题"""
        total_se = 0.0
        n_samples = len(test_preds)

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)

            # 获取当前批次
            batch_preds = test_preds[i:end_idx]
            batch_actuals = test_actuals[i:end_idx]

            # 确保是列表或一维数组
            if isinstance(batch_preds, list):
                batch_preds = np.array(batch_preds, dtype=np.float32)
            if isinstance(batch_actuals, list):
                batch_actuals = np.array(batch_actuals, dtype=np.float32)

            # 展平
            batch_preds = batch_preds.ravel()
            batch_actuals = batch_actuals.ravel()

            # 计算平方误差
            se = (batch_preds - batch_actuals) ** 2
            total_se += np.sum(se)

        return np.sqrt(total_se / n_samples)

    def run_cross_validation(self, rank=50, lambda_reg=80, model_name="SoftImpute"):
        """运行交叉验证"""
        splits = self.data_loader.create_cross_validation_splits(n_folds=self.n_folds)

        # 初始化结果存储
        self.results = {
            'test_rmses': [],
            'train_rmses': [],
            'convergence': [],
            'train_times': []
        }

        print(f"开始{self.n_folds}折交叉验证...")
        print(f"矩阵秩: {rank}, λ: {lambda_reg}")
        print("=" * 60)

        for fold_idx, (train_indices, test_indices) in enumerate(splits, 1):
            print(f"\n第{fold_idx}折:")
            print(f"训练集大小: {len(train_indices)}, 测试集大小: {len(test_indices)}")

            # 创建训练和测试数据
            R_train, test_data = self.create_train_test_matrices(train_indices, test_indices)

            # 初始化模型
            model = EfficientSoftImpute(
                rank=rank,
                lambda_reg=lambda_reg,
                max_iter=15,
                tol=1e-4,
                randomized=True,
                verbose=True
            )

            # 训练模型
            start_time = time.time()
            model.fit(R_train)
            train_time = time.time() - start_time

            # 批量预测和评估（避免内存问题）
            print(f"开始预测 {len(test_data)} 个测试样本...")
            batch_size = 50000  # 每批处理50000个样本

            test_preds = []
            test_actuals = []

            for i in range(0, len(test_data), batch_size):
                end_idx = min(i + batch_size, len(test_data))
                batch_data = test_data[i:end_idx]

                # 提取用户和物品索引
                batch_users = [u for u, _, _ in batch_data]
                batch_items = [m for _, m, _ in batch_data]
                batch_ratings = [r for _, _, r in batch_data]

                # 批量预测
                batch_preds = model.predict(batch_users, batch_items)

                # 确保预测结果是一维的
                if hasattr(batch_preds, 'ndim') and batch_preds.ndim > 1:
                    batch_preds = batch_preds.flatten()

                # 收集结果
                if isinstance(batch_preds, np.ndarray):
                    test_preds.extend(batch_preds.tolist())
                else:
                    test_preds.extend([float(p) for p in batch_preds])

                test_actuals.extend(batch_ratings)

                if (i // batch_size) % 10 == 0 or end_idx == len(test_data):
                    print(f"  已处理 {end_idx}/{len(test_data)} 个样本...")

            print("计算RMSE...")

            # 使用安全的方式计算RMSE
            test_rmse = self.safe_compute_rmse(test_preds, test_actuals)

            # 获取训练误差
            if hasattr(model, 'iter_history') and model.iter_history:
                # 取最后一次迭代的训练MSE（中心化后的）
                train_mse = model.iter_history[-1][1] if model.iter_history else 0
                train_rmse = np.sqrt(train_mse) if train_mse > 0 else 0
            else:
                train_rmse = 0

            # 存储结果
            self.results['test_rmses'].append(test_rmse)
            self.results['train_rmses'].append(train_rmse)
            self.results['convergence'].append(model.iter_history if hasattr(model, 'iter_history') else [])
            self.results['train_times'].append(train_time)

            print(f"  测试RMSE: {test_rmse:.4f}, 训练RMSE: {train_rmse:.4f}, 训练时间: {train_time:.2f}秒")

            # 保存历史记录（类似第一个文件的做法）
            final_metric = test_rmse  # 使用测试RMSE作为最终指标
            saved_path = self._save_history(model_name, fold_idx,
                                            model.iter_history, final_metric, train_time)
            print(f"  历史记录保存到: {saved_path}")

        return self.results

    def print_results(self):
        """打印结果"""
        print("\n" + "=" * 60)
        print("Soft-Impute交叉验证结果汇总")
        print("=" * 60)

        test_rmses = self.results['test_rmses']
        train_times = self.results['train_times']

        if len(test_rmses) > 0:
            mean_rmse = np.mean(test_rmses)
            std_rmse = np.std(test_rmses)
            mean_time = np.mean(train_times)

            print(f"平均测试RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            print(f"平均训练时间: {mean_time:.2f}秒")
            print(f"各折RMSE: {[f'{rmse:.4f}' for rmse in test_rmses]}")
            print(f"各折训练时间: {[f'{t:.2f}' for t in train_times]}秒")

    def plot_convergence(self, fold_idx=0):
        """绘制收敛曲线"""
        try:
            if not self.results.get('convergence'):
                print("没有收敛数据可绘制")
                return

            convergence_data = self.results['convergence']

            # 绘制指定折的收敛曲线
            if fold_idx < len(convergence_data) and convergence_data[fold_idx]:
                history = convergence_data[fold_idx]
                if history:
                    plt.figure(figsize=(10, 6))

                    # 提取时间和MSE
                    times = [t for t, _ in history]
                    mses = [mse for _, mse in history]

                    plt.plot(times, mses, 'b-o', linewidth=2, markersize=6)
                    plt.xlabel('时间 (秒)', fontsize=12)
                    plt.ylabel('中心化MSE', fontsize=12)
                    plt.title(f'Soft-Impute 收敛曲线 (第{fold_idx + 1}折)', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # 保存图片
                    plt.savefig('softimpute_convergence.png', dpi=150, bbox_inches='tight')
                    plt.show()

                    print("收敛曲线已保存为 'softimpute_convergence.png'")

            # 绘制所有折的测试RMSE比较
            plt.figure(figsize=(10, 6))
            folds = range(1, len(self.results['test_rmses']) + 1)
            plt.bar(folds, self.results['test_rmses'], color='skyblue', edgecolor='black')
            plt.xlabel('折数', fontsize=12)
            plt.ylabel('测试RMSE', fontsize=12)
            plt.title('Soft-Impute 各折测试RMSE比较', fontsize=14)
            plt.xticks(folds)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            # 添加数值标签
            for i, v in enumerate(self.results['test_rmses']):
                plt.text(i + 1, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

            plt.savefig('softimpute_rmse_by_fold.png', dpi=150, bbox_inches='tight')
            plt.show()

            print("RMSE比较图已保存为 'softimpute_rmse_by_fold.png'")

        except Exception as e:
            print(f"绘图错误: {e}")
            import traceback
            traceback.print_exc()


# 4. 运行不同组合
def run_experiments(data_loader):
    evaluator = SoftImputeExperiment(data_loader, n_folds=5, history_dir="history")

    results = []
    # Soft-Impute组合
    configs = [
        {"name": "SoftImpute(rank=10)", "rank": 10, "lambda_reg": 60},
        {"name": "SoftImpute(rank=50)", "rank": 50, "lambda_reg": 60},
        {"name": "SoftImpute(rank=100)", "rank": 100, "lambda_reg": 60},
        {"name": "SoftImpute(rank=100_2)", "rank": 100, "lambda_reg": 0.1},
    ]

    for cfg in configs:
        print("\n" + "-" * 60)
        print(f"运行 {cfg['name']}")

        try:
            # 运行交叉验证
            evaluator.run_cross_validation(
                rank=cfg['rank'],
                lambda_reg=cfg['lambda_reg'],
                model_name=cfg['name']
            )

            # 收集结果
            test_rmses = evaluator.results['test_rmses']
            train_times = evaluator.results['train_times']

            if len(test_rmses) > 0:
                avg_rmse = np.mean(test_rmses)
                avg_time = np.mean(train_times)
            else:
                avg_rmse = float('inf')
                avg_time = float('inf')

            results.append({
                "Model": cfg['name'],
                "Avg RMSE": avg_rmse,
                "Avg Time (s)": avg_time,
                "Fold 1 RMSE": test_rmses[0] if len(test_rmses) > 0 else None,
                "Fold 2 RMSE": test_rmses[1] if len(test_rmses) > 1 else None,
                "Fold 3 RMSE": test_rmses[2] if len(test_rmses) > 2 else None,
                "Fold 4 RMSE": test_rmses[3] if len(test_rmses) > 3 else None,
                "Fold 5 RMSE": test_rmses[4] if len(test_rmses) > 4 else None,
            })

            # 显示当前组合的结果
            evaluator.print_results()

        except Exception as e:
            print(f"配置 {cfg['name']} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    print("\n最终结果汇总:")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 保存结果到CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"softimpute_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\n结果已保存到 '{csv_filename}'")

    return results_df


# 主函数
def main():
    """主函数"""
    print("=" * 60)
    print("高效的Soft-Impute算法实现")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")

    # 尝试不同的数据文件路径
    possible_paths = [
        'ratings.dat',  # 当前目录
    ]

    data_loader = None
    data_file = None

    for path in possible_paths:
        if os.path.exists(path):
            print(f"找到数据文件: {path}")
            data_file = path
            break

    if data_file is None:
        print("未找到数据文件，请输入数据文件路径:")
        data_file = input().strip()
        if not os.path.exists(data_file):
            print(f"文件 {data_file} 不存在，使用测试数据")
            data_file = None

    if data_file:
        data_loader = MovieLensData(data_file)
        data_loader.load_ratings()
    else:
        print("使用测试数据...")
        data_loader = MovieLensData(None)
        data_loader.create_test_data()

    print(f"\n数据集信息:")
    print(f"  用户数: {data_loader.n_users}")
    print(f"  电影数: {data_loader.n_movies}")
    print(f"  评分总数: {len(data_loader.ratings_list)}")

    # 2. 运行实验
    print("\n2. 运行不同配置的实验...")
    results_df = run_experiments(data_loader)

    # 3. 绘制第一折的收敛曲线
    print("\n3. 生成可视化图表...")
    try:
        # 加载第一折的历史数据来绘制收敛曲线
        history_files = [f for f in os.listdir("history") if f.endswith("_fold1_history.npz")]
        if history_files:
            # 使用第一个配置的第一折历史
            history_file = history_files[0]
            data = np.load(os.path.join("history", history_file))
            if 'iter_history' in data:
                history = data['iter_history']
                if history.shape[0] > 0:
                    plt.figure(figsize=(10, 6))
                    plt.plot(history[:, 0], history[:, 1], 'b-o', linewidth=2, markersize=6)
                    plt.xlabel('时间 (秒)', fontsize=12)
                    plt.ylabel('中心化MSE', fontsize=12)
                    plt.title(f'Soft-Impute 收敛曲线 (第一折)', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('final_convergence_curve.png', dpi=150, bbox_inches='tight')
                    plt.show()
                    print("收敛曲线已保存为 'final_convergence_curve.png'")
    except Exception as e:
        print(f"绘制收敛曲线时出错: {e}")

    print("\n实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子以确保可重复性
    main()

# datasets/synthetic.py
import numpy as np
from config import MALIBOConfig

class ForresterHeterogeneousBenchmark:
    def __init__(self, n_points=500):
        self.domain = np.linspace(0, 1, n_points).reshape(-1, 1)

    def evaluate_target(self, x, scale=2.5, shift=0.3, noise_std=MALIBOConfig.NOISE_STD):
        """Hàm Target Task bị Dịch chuyển Phân phối (Outlier / Hard Task)"""
        # Áp dụng scale & shift khác biệt so với tập Meta-training
        y_true = scale * ((6 * (x - shift) - 2) ** 2) * np.sin(12 * (x - shift) - 4)
        noise = np.random.normal(0, noise_std, size=x.shape)
        return y_true + noise

    def get_meta_tasks(self, n_tasks=MALIBOConfig.N_META_TASKS, samples_per_task=80):
        """Tạo tập các Meta-tasks cũ với biên độ biến thiên nhẹ"""
        tasks = []
        for _ in range(n_tasks):
            scale = np.random.uniform(0.8, 1.2)
            shift = np.random.uniform(-0.1, 0.1)
            
            X_task = np.random.uniform(0, 1, size=(samples_per_task, 1))
            Y_task = scale * ((6 * (X_task - shift) - 2) ** 2) * np.sin(12 * (X_task - shift) - 4)
            tasks.append((X_task, Y_task))
        return tasks
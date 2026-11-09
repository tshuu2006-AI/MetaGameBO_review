import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class MALIBOAdaptor:
    def __init__(self, meta_model, input_dim=1, embed_dim=4, gamma=0.25):
        self.model = meta_model
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.boost_model = GradientBoostingRegressor(n_estimators=15, max_depth=3)
        self.is_boosted = False

    def fit_laplace_posterior(self, X_obs, y_obs, steps=80, lr=0.01):
        """Tìm z_MAP và ma trận hiệp biến Laplace Σ_N chuẩn hóa PSD"""
        threshold = np.quantile(y_obs, self.gamma)
        c_labels = (y_obs <= threshold).astype(np.float32)
        
        X_t = torch.tensor(X_obs, dtype=torch.float32)
        c_t = torch.tensor(c_labels, dtype=torch.float32).reshape(-1, 1)

        z_param = torch.zeros(1, self.embed_dim, requires_grad=True)
        opt = optim.Adam([z_param], lr=lr)
        bce = nn.BCELoss()

        # 1. Tối ưu z_MAP
        for _ in range(steps):
            opt.zero_grad()
            pred = self.model(X_t, z_param)
            loss = bce(pred, c_t) + 0.5 * torch.sum(z_param ** 2)
            loss.backward()
            opt.step()

        z_map = z_param.detach().clone()

        # 2. Tính Ma trận Hessian
        def loss_fn(z):
            pred = self.model(X_t, z)
            return bce(pred, c_t) + 0.5 * torch.sum(z ** 2)

        hessian = torch.autograd.functional.hessian(loss_fn, z_map).squeeze()
        
        if hessian.dim() == 0:
            hessian = hessian.unsqueeze(0).unsqueeze(0)
        elif hessian.dim() == 1:
            hessian = torch.diag(hessian)

        # Thêm Jitter đường chéo
        hessian += torch.eye(self.embed_dim) * 1e-3
        cov_matrix = torch.inverse(hessian).detach().numpy()

        # --- SỬA TẠI ĐÂY: Chuẩn hóa Covariance về dạng Symmetric Positive Semi-Definite (PSD) ---
        # A. Ép ma trận đối xứng hoàn hảo
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
        
        # B. Phân rã trị riêng và kẹp trị riêng âm thành dương min (1e-4)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, 1e-4)
        
        # C. Tái tạo ma trận Hiệp biến sạch
        clean_cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return z_map.numpy().squeeze(), clean_cov_matrix

    def sample_thompson(self, z_map, cov_matrix):
        """Rút mẫu z_hat với check_valid='ignore' an toàn tuyệt đối"""
        # Sử dụng check_valid='ignore' để tắt hoàn toàn RuntimeWarning dư thừa của numpy
        z_sample = np.random.multivariate_normal(z_map, cov_matrix, check_valid='ignore')
        return torch.tensor(z_sample, dtype=torch.float32).unsqueeze(0)

    def fit_residual_boosting(self, X_obs, y_obs, z_hat_tensor):
        """Fit mô hình Gradient Boosting trên phần dư"""
        if len(X_obs) < 4:
            return
        X_t = torch.tensor(X_obs, dtype=torch.float32)
        with torch.no_grad():
            meta_preds = self.model(X_t, z_hat_tensor).numpy().squeeze()
        
        residuals = (-y_obs.squeeze()) - meta_preds
        self.boost_model.fit(X_obs, residuals)
        self.is_boosted = True

    def predict_acquisition(self, X_candidates, z_hat_tensor):
        """Tính α(x) = f_meta(x) + f_boost(x)"""
        X_t = torch.tensor(X_candidates, dtype=torch.float32)
        with torch.no_grad():
            meta_scores = self.model(X_t, z_hat_tensor).numpy().squeeze()

        if self.is_boosted:
            boost_scores = self.boost_model.predict(X_candidates)
            return meta_scores + boost_scores
        return meta_scores
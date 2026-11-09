import numpy as np

def preprocess_task_data(X, Y, gamma=0.25):
    """
    Tiền xử lý dữ liệu task:
    1. Chuẩn hóa Min-Max cho X về miền [0, 1]
    2. Gán nhãn nhị phân dựa trên ngưỡng Quantile gamma (Density Ratio Trick)
    """
    # Chuẩn hóa X
    X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    
    # Gán nhãn nhị phân cho Y
    y_threshold = np.quantile(Y, gamma)
    labels = (Y <= y_threshold).astype(np.float32)  # 1 cho Top 25% tốt nhất, 0 cho lại
    
    return X_norm, Y, labels, y_threshold
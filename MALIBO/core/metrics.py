import numpy as np

def compute_normalized_regret(evaluated_y, f_min, f_max):
    """
    Normalized Regret = (min(y_eval) - f_min) / (f_max - f_min)
    """
    best_y_so_far = np.minimum.accumulate(evaluated_y)
    regret = (best_y_so_far - f_min) / (f_max - f_min + 1e-8)
    return regret

def compute_cvar(regrets_across_runs, alpha=0.05):
    """
    CVaR_alpha: Giá trị Regret trung bình của (1 - alpha) % các lần chạy tệ nhất
    """
    # Lấy giá trị regret ở bước cuối cùng
    final_regrets = regrets_across_runs[:, -1]
    sorted_regrets = np.sort(final_regrets)
    cutoff_index = int(np.ceil((1 - alpha) * len(sorted_regrets)))
    tail_regrets = sorted_regrets[cutoff_index:]
    return np.mean(tail_regrets)
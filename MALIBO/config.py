# config.py
class MALIBOConfig:
    # ---------------------------------------------------------
    # 1. Hyperparameters cho Likelihood-Free Classification
    # ---------------------------------------------------------
    QUANTILE_GAMMA = 0.25      # Ngưỡng γ để gán nhãn nhị phân Top 25% (Density Ratio Trick)[cite: 4]
    INPUT_DIM = 1              # Số chiều không gian tìm kiếm x (1D với Forrester)
    EMBED_DIM = 4              # Số chiều vector ẩn z_tau (Task Embedding)[cite: 3, 5]
    HIDDEN_DIM = 64            # Số nơ-ron lớp ẩn của MetaNet

    # ---------------------------------------------------------
    # 2. Hyperparameters cho Offline Meta-Training (Phase 1)
    # ---------------------------------------------------------
    META_EPOCHS = 300          # Số epoch huấn luyện Mạng Meta
    META_LR = 0.001            # Tốc độ học của Adam Optimizer cho trọng số theta
    KL_BETA = 0.01             # Hệ số phạt KL Divergence: β * KL(q(z) || N(0, I))[cite: 5]
    N_META_TASKS = 50          # Số lượng task cũ dùng để Meta-train

    # ---------------------------------------------------------
    # 3. Hyperparameters cho Target Adaptation (Phase 2)
    # ---------------------------------------------------------
    ADAPT_STEPS = 100          # Số bước Gradient Descent để tìm z_MAP trên task mới[cite: 5]
    ADAPT_LR = 0.01            # Tốc độ học khi tìm z_MAP[cite: 5]
    JITTER = 1e-4              # Jitter đường chéo ổn định ma trận Hessian H[cite: 5]
    
    # ---------------------------------------------------------
    # 4. Hyperparameters cho Residual Gradient Boosting
    # ---------------------------------------------------------
    BOOST_N_ESTIMATORS = 10    # Số cây quyết định trong Gradient Boosting[cite: 5]
    BOOST_MAX_DEPTH = 3        # Độ sâu tối đa của cây học phần dư[cite: 5]

    # ---------------------------------------------------------
    # 5. Experimental Setup & Benchmarking
    # ---------------------------------------------------------
    N_RUNS = 100               # Số lần chạy độc lập lấy trung bình
    EVAL_BUDGET = 10           # Ngân sách đánh giá N = 10 steps (khớp thiết lập Bảng 1)
    INITIAL_SEEDS = 3          # Số điểm thử ngẫu nhiên ban đầu
    CVAR_ALPHA = 0.05          # Mức tin cậy α = 0.05 tính CVaR (Tail risk 5% tệ nhất)[cite: 3, 6]
    
    # Thiết lập Distribution Shift (Tạo độ khó thực tế theo bài báo)
    NOISE_STD = 0.1            # Nhiễu quan sát[cite: 1]
    SCALE_RANGE = (0.2, 3.0)   # Biên độ co giãn ngẫu nhiên giữa các task[cite: 1]
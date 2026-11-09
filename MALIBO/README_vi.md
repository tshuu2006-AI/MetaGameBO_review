# MALIBO + PFN baselines cho thực nghiệm Forrester

Workspace này tách **hai baseline** cần đối chiếu với Table 1 trong bài MetaGameBO:

- `MALIBO`: gọi implementation chính thức của Pan et al. (ICML 2024), không dùng MLP rút gọn đang có trong archive `MALIBO.rar`.
- `PFN`: gọi implementation PFNs4BO chính thức của Müller et al. (ICML 2023). Bài AAAI-26 ghi `PFN`; `PNF` có vẻ là đảo chữ.

Đây là lớp tích hợp/thực nghiệm mỏng. Mã upstream của MALIBO và PFNs4BO được cài riêng theo license của từng dự án, không được chép lẫn vào source workspace này.

## 1. Cấu trúc

```text
MALIBO_PFN_refactor/
├── benchmarks/forrester.py
├── configs/experiment.py
├── docs/AUDIT_VS_PAPER.md
├── experiments/metrics.py
├── experiments/run_comparison.py
├── models/malibo_adapter.py
├── models/pfn_adapter.py
├── results/
├── tests/
├── main.py
├── requirements.txt
├── requirements-malibo.txt
└── requirements-pfn.txt
```

## 2. Cài đặt

Khuyến nghị Python 3.11. MALIBO upstream cung cấp `environment.yml` với các phiên bản đã dùng trong implementation của họ (ví dụ `blackboxopt==5.0.4`, `parameterspace==0.10.0`, `torch==2.4.0`, `scikit-learn==1.5.1`).

```bash
# Tạo môi trường cơ sở
conda create -n malibo-pfn python=3.11 -y
conda activate malibo-pfn

# Cài các dependency của MALIBO
pip install -r requirements-malibo.txt

# Lấy source chính thức MALIBO vào third_party
mkdir -p third_party
git clone https://github.com/boschresearch/meta-learning-likelihood-free-bayesian-optimization.git \
  third_party/meta-learning-likelihood-free-bayesian-optimization

# Cài PFNs4BO (nếu pip resolver làm thay đổi torch/scipy, nên dùng env riêng cho PFN)
pip install -r requirements-pfn.txt
```

PFNs4BO README hiện khuyến nghị `pip install pfns4bo` và sử dụng model đã train sẵn. Repository PFNs4BO lưu ý code cũ không còn được maintain tích cực; hãy giữ nguyên phiên bản package/model trong log thực nghiệm.

## 3. Chạy thử

```bash
# Smoke test logic thuần numpy
python -m pytest -q

# Chạy cả 2 baseline
python main.py --methods malibo pfn --runs 100 --steps 16 --initial-points 4

# Hoặc chạy từng baseline trong các môi trường tách biệt
python main.py --methods malibo --runs 100 --steps 16 --initial-points 4
python main.py --methods pfn --runs 100 --steps 16 --initial-points 4
```

Mặc định MALIBO source được tìm ở `third_party/meta-learning-likelihood-free-bayesian-optimization`. Có thể truyền đường dẫn khác:

```bash
python main.py --methods malibo --malibo-repo /path/to/official-malibo
```

Kết quả được lưu trong `results/forrester/`:

- `summary.csv`: mean normalized regret, standard error, CVaR95.
- `<method>_curves.npz`: đường cong regret của từng run.
- `config.json`: cấu hình đã chạy.

## 4. Thiết lập baseline trong workspace

### MALIBO

Adapter dùng API `MALIBO` upstream (`meta_fit`, `observe_and_suggest`) và giữ meta-model đã học cố định giữa các target run. Mỗi target run tạo optimizer mới để Bayesian logistic regression/adaptation không bị rò rỉ giữa các task.

Cấu hình khởi đầu lấy từ `run_forrester.py` chính thức của MALIBO: 256 meta-task, 32 observations/task, 5 layers, 50 learned features, 64 hidden units, 2048 epochs, batch size 256, 16 BO iterations. `gamma=1/3` và candidate pool 5120 theo mặc định của class MALIBO.

### PFN

Adapter sử dụng `TransformerBOMethod` trong `pfns4bo.scripts.acquisition_functions`. Mặc định model là `hebo_plus_model`; có thể chọn `bnn_model` bằng `--pfn-model bnn`. PFNs4BO yêu cầu `X_obs` và candidate `X_pending` được chuẩn hóa trong [0, 1].

**Lưu ý quan trọng:** bài AAAI-26 không nêu trong phần main paper model PFN cụ thể (HEBO+ hay BNN), số điểm warm-start, số candidate tối ưu acquisition, hay toàn bộ cấu hình Forrester. Vì vậy, kết quả của workspace này là baseline có cấu hình công khai, không được tuyên bố là tái lập chính xác Table 1 cho tới khi đối chiếu Appendix/supplementary của bài báo.

## 5. Giả định đang được công khai

1. Forrester task family dùng công thức/task descriptors trong benchmark chính thức MALIBO: `f(x) = a(6x-2)^2 sin(12x-4) + b(x-0.5) - c`, với `a∈[0.2,3]`, `b∈[-5,15]`, `c∈[-5,5]`.
2. Mỗi task dùng 4 initial points chung cho cả hai baseline, sau đó 16 đề xuất BO. Đây là lựa chọn để đưa hai optimizer vào cùng giao thức; bài AAAI-26 cần Appendix để xác nhận ngân sách/initialization chính xác.
3. Objective mặc định không nhiễu. Thêm noise/distribution shift chỉ sau khi có thông số chính thức từ Appendix.
4. Candidate pool chung 5120 điểm mỗi vòng, lấy mẫu đều trên [0,1], để tránh so sánh hai thuật toán trên hai không gian ứng viên khác nhau.
5. CVaR được báo cáo là mean của 5% run có regret cuối lớn nhất (`CVaR95`, tức upper-tail 5%).

Không hard-code kết quả kỳ vọng từ Table 1 vào chương trình. Chương trình phải báo số đo thực nghiệm mới từ log.

## 6. Nguồn thuật toán

- MALIBO (Pan et al., ICML 2024): https://github.com/boschresearch/meta-learning-likelihood-free-bayesian-optimization
- PFNs4BO (Müller et al., ICML 2023): https://github.com/automl/PFNs4BO
- Bài MetaGameBO (AAAI-26): https://ojs.aaai.org/index.php/AAAI/article/view/40995

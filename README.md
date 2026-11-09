# MetaGameBO reproducibility study

Bộ mã PyTorch độc lập để so sánh MALIBO-style likelihood-free meta-BO, PFN
in-context surrogate, và MetaGameBO-style robust meta-learning.

> Đây là **reference reimplementation** phục vụ kiểm thử độc lập, không phải bản
> sao mã AGPL chính thức của tác giả. Các lựa chọn thực nghiệm (kiến trúc, lịch
> tối ưu, prior PFN, siêu tham số k-DPP) được đặt rõ trong mã/CLI.

## Cài đặt và chạy

```powershell
python -m pip install -r requirements.txt

# Smoke test nhanh
python main.py --quick

# Protocol đầy đủ: 100 seed, 50 BO step, 512 mẫu/task
python main.py --runs 100 --steps 50 --samples-per-task 512

# Kiểm thử
python -m pytest -q
```

Kết quả được ghi vào `results/forrester/`: `summary.csv`, ma trận curve `.npy`,
`config.json`, và `convergence.png`.

## Thiết kế

- `benchmarks/`: Forrester dị thể có rare tail shocks; `BaseBenchmark` và
  `MockBenchmark` là hợp đồng tích hợp HPOBench/NASBench201/HPO-B.
- `models/malibo.py`: embedding biến phân, KL, Laplace posterior, Thompson
  sampling, residual gradient boosting bằng decision stumps.
- `models/pfn.py`: Transformer prior-fitted; target inference chỉ condition lên
  context, hoàn toàn không gradient update.
- `models/metagamebo.py`: hard-task selection theo CVaR và k-DPP với kernel
  quality-diversity ở cấp sample.
- `experiment.py`: leave-one-task-out, seed 0..99, báo cáo mean +/- std, CVaR
  upper tail và convergence curve.

Normalized regret dùng đúng `(best - f_min) / (f_max - f_min)`. Vì observation
có noise/tail shock, regret đo được đôi lúc có thể nhỏ hơn 0; mã không clip để
không che giấu sai lệch thống kê.

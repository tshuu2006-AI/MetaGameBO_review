# LFBO + GP — Forrester experiments

Phần thực nghiệm của Phước: so sánh **LFBO-EI** (Random Forest 1.000 cây) và **GP regression** (Constant × Matérn 5/2) trên Forrester. Pipeline chính là unified Track B: 100 family độc lập, seeds 44–143, mỗi family có 200 tasks và 32 quan sát/task, chia raw train/validation 26/6 trước augmentation.

Metric là **Weighted BCE theo task**, chia cho số augmented validation rows; CVaR₀.₈ lấy 39 task losses lớn nhất trong mỗi family. Bảng cuối lấy mean ± sample SE trên 100 family metrics. GP dùng `q = EI_tau / (EI_tau + U_train)`; đây là evaluator suy diễn của nhóm, **chưa được tác giả xác minh**. Kết quả mang nhãn `PAPER_INSPIRED_INDEPENDENT_REPRODUCTION / NOT_PAPER_EQUIVALENT`, không phải tái lập chính xác Table 1.

## Cài đặt Ubuntu

Runtime được khóa ở **Python 3.12.14**; kiểm tra `python3.12 --version` trước khi tạo môi trường. Dependencies CPU nằm trong `requirements.txt`; không cần checkpoint hoặc dịch vụ bên ngoài.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 -B main.py --mode preflight
```

## Chạy bằng terminal

Chọn run ID mới để giữ nguyên kết quả trước. Không có thực nghiệm nào tự chạy khi gọi help hoặc preflight.

```bash
python3 -B main.py --help
python3 -B -u main.py --mode smoke --run-id lfbo-gp-smoke-user --workers 1
python3 -B -u main.py --mode family --run-id lfbo-gp-family200-user --workers 1

# 100 family × 200 tasks: 20.000 fits cho mỗi phương pháp.
python3 -B -u main.py --mode full --run-id lfbo-gp-full100-user --workers 2 \
  --confirm-full --acknowledge-independent-protocol

python3 -B -u main.py --mode resume --run-id lfbo-gp-full100-user --workers 2 \
  --confirm-full --acknowledge-independent-protocol
```

## Xem kết quả đã lưu

```bash
python3 -S -B main.py --mode show --run-id lfbo-gp-full100-user
python3 -S -B main.py --mode compare --run-id lfbo-gp-full100-user
python3 -S -B main.py --mode verify --run-id lfbo-gp-full100-user
```

Các lệnh này chỉ đọc và kiểm tra dữ liệu, không training/inference/objective. Thiếu dữ liệu sẽ báo lỗi, không sinh số liệu mới. Kết quả lưu local tại `results/lfbo_gp_unified/<run-id>/`: config/source snapshots, seeds, raw observations/predictions, task/family metrics, CSV, JSON, logs và biểu đồ. Báo cáo trung gian và dữ liệu thực nghiệm không đưa lên Git.

Bản code xuất bản dùng `configs/lfbo_gp_unified/protocol_v2.json`. Các tham số khoa học, seed và split giữ nguyên v1; đường dẫn nguồn và source lineage được tách lại cho clone mới. Kết quả v1 local vẫn được xem/verify bằng snapshot gốc, ví dụ:

```bash
python3 -S -B main.py --mode compare --run-id unified-b-full100-20261114
```

V1 chỉ đọc dưới code v2. Muốn resume v1 phải dùng đúng code/runtime v1; không sửa protocol lock cũ để vượt kiểm tra. Clone mới không kèm dữ liệu local này.

## Kiểm thử

```bash
.venv/bin/python -B -m pytest
```

Tests dùng fixture nhỏ, không chạy benchmark full100. Kiểm tra số học PyTorch/source AST là tùy chọn, trong môi trường CPU Python 3.11 riêng:

```bash
python3.11 -m venv .venv-torch
.venv-torch/bin/python -m pip install -r requirements-torch-check.txt
.venv-torch/bin/python -B -m pytest tests/test_lfbo_gp_unified_torch.py
```

Nguồn tham chiếu và licenses giữ tại `vendor/`; xem `NOTICE` và `LICENSE`. Email, ảnh trao đổi và báo cáo kiểm toán cá nhân chỉ giữ local.

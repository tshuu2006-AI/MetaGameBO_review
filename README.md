# LFBO và GP trên Forrester

Thực nghiệm hai thuật toán Bayesian Optimization là LFBO và GP trên họ hàm Forrester, đối chiếu với kết quả trong bài báo MetaGameBO.

## Kết quả

Chạy 100 bộ dữ liệu độc lập với seed từ 44 đến 143. Mỗi bộ gồm 200 tasks, mỗi task có 32 quan sát và được chia thành 26 điểm train, 6 điểm validation trước khi tạo các dòng phân loại. Tổng cộng 20.000 tasks cho mỗi thuật toán.

| Thuật toán | Paper: Average Loss ↓ | Thực nghiệm: Weighted BCE ± SE ↓ | Paper: CVaR ↓ | Thực nghiệm: CVaR₀.₈ ± SE ↓ |
|---|---:|---:|---:|---:|
| LFBO | 0.5823 | 0.523973 ± 0.027310 | 0.9876 | 1.303971 ± 0.138380 |
| GP | 0.7891 | 0.328993 ± 0.001370 | 1.3245 | 0.600549 ± 0.002292 |

Số liệu paper lấy từ Table 1; kết quả thực nghiệm lấy từ [summary.json](results/published/lfbo-gp-full100-20261114/summary.json). Phần ± là sai số chuẩn của trung bình trên 100 bộ dữ liệu độc lập.

Bảng sử dụng classifier loss, không phải normalized regret sau BO. Cách chia dữ liệu và cấu hình đánh giá GP còn khác paper, nên các số liệu trên chưa thể xem là tái lập hoàn toàn Table 1.

## Cài đặt và chạy

Môi trường: Ubuntu, Python 3.12.14. Các thư viện được ghi trong `requirements.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 main.py --mode preflight

# Chạy thử trên 3 tasks
python3 -u main.py --mode smoke --run-id smoke-01 --workers 1

# Chạy một bộ 200 tasks
python3 -u main.py --mode family --run-id family-01 --workers 1

# Chạy đủ 100 bộ dữ liệu
python3 -u main.py --mode full --run-id full100-01 --workers 2 \
  --confirm-full --acknowledge-independent-protocol

# Tiếp tục lần chạy đang dở
python3 -u main.py --mode resume --run-id full100-01 --workers 2 \
  --confirm-full --acknowledge-independent-protocol
```

Dùng run ID mới cho mỗi lần chạy. Mặc định lưu mô hình LFBO và GP của task đầu tiên để kiểm tra lại dự đoán. Có thể dùng `--checkpoints all` để lưu mọi mô hình hoặc `--checkpoints none` để tắt.

## Thuật toán và cách tính metric

- **LFBO:** Random Forest 1.000 cây, EI augmentation, gamma = 1/3. Ngưỡng tau và weights được tính từ dữ liệu train.
- **GP:** Gaussian process với kernel Constant × Matérn 5/2. Xác suất dùng để tính classifier loss là `q = EI_tau / (EI_tau + U_train)`. Đây là cách đánh giá do nhóm lựa chọn, chưa được tác giả xác nhận.
- **Weighted BCE theo task:** tổng `weight × BCE` chia cho số dòng validation sau augmentation, không chia cho tổng weights.
- **CVaR₀.₈:** trung bình các task losses lớn nhất trong từng bộ dữ liệu, với `k = max(1, int(N * (1 - alpha)))`. Khi N = 200, cách tính float của Python cho k = 39. Sau đó lấy trung bình và SE trên 100 bộ.

Cấu hình nằm trong `configs/lfbo_gp_unified/`. Mã nguồn tham chiếu và giấy phép nằm trong `vendor/`, `NOTICE` và `LICENSE`.

## Dữ liệu kết quả

[Kết quả đầy đủ ở dạng gọn](results/published/lfbo-gp-full100-20261114/) khoảng 29,5 MB, gồm:

| File | Nội dung |
|---|---|
| `summary.json`, `run_metrics.json` | Bảng tổng hợp và metric của từng bộ dữ liệu |
| `task_losses.csv` | 40.000 task losses của LFBO và GP |
| `paired_task_differences.csv` | Chênh lệch loss giữa hai thuật toán trên cùng task |
| `evidence/meta_dataset.npz` | 640.000 quan sát gốc, chỉ số train/validation và các hệ số tính từ train |
| `evidence/classifier_predictions.npz` | Xác suất, posterior GP, logits, labels, weights và loss trên validation |
| `config.json`, `environment.json`, `logs/` | Cấu hình, môi trường và logs |
| `publication_manifest.json`, `evidence/evidence_manifest.json` | Nguồn file và SHA256 để đối chiếu |

Run full100 gốc không lưu checkpoint mô hình. Bộ [smoke 3 tasks](results/published/lfbo-gp-evidence-smoke-20261118/) có hai checkpoint đại diện được lưu khi fit; dự đoán của cả ba tasks khớp các tasks tương ứng trong run full100.

Raw archive dạng JSON và source snapshot khoảng 1,45 GB được giữ local trong `results/lfbo_gp_unified/`. Dữ liệu NPZ trên GitHub dùng để kiểm tra kết quả, không dùng để resume.

## Kiểm tra lại kết quả

Kiểm tra dữ liệu full100 đã có trên GitHub:

```bash
python3 main.py --mode verify-evidence \
  --evidence-dir results/published/lfbo-gp-full100-20261114/evidence
```

Chạy lại dự đoán từ hai checkpoint smoke đã lưu:

```bash
python3 main.py --mode replay-checkpoints \
  --evidence-dir results/published/lfbo-gp-evidence-smoke-20261118
```

`verify-evidence` tính lại metric từ dữ liệu đã lưu. `replay-checkpoints` chỉ chạy inference từ mô hình có sẵn. Cả hai đều không training lại.

Để xuất bằng chứng từ một lần chạy mới, chọn thư mục đích chưa tồn tại:

```bash
python3 main.py --mode export-evidence --run-id smoke-01 \
  --evidence-dir results/evidence/smoke-01
```

Nếu máy còn raw archive đầy đủ, có thể dùng `--mode show`, `--mode compare` hoặc `--mode verify` cùng `--run-id` để xem và kiểm tra lần chạy đó.

## Kiểm thử

```bash
.venv/bin/python -m pytest
```

Đã kiểm tra 110 tests chính và 6 tests đối chiếu PyTorch; smoke thật và replay checkpoint đều chạy thành công. Logs kiểm thử nằm trong [thư mục smoke](results/published/lfbo-gp-evidence-smoke-20261118/logs/).

Các tests PyTorch chạy riêng với Python 3.11:

```bash
python3.11 -m venv .venv-torch
.venv-torch/bin/python -m pip install -r requirements-torch-check.txt
.venv-torch/bin/python -m pytest tests/test_lfbo_gp_unified_torch.py
```

# LFBO và GP trên Forrester

Phần thực nghiệm của Phước trong MetaGameBO review: **LFBO-EI (Random Forest 1.000 cây)** và **GP regression (Constant × Matérn 5/2)**. Pipeline chính là unified Track B, đánh giá classifier loss trên dữ liệu held-out.

## Kết quả: Paper và thực nghiệm của Phước

Run **`unified-b-full100-20261114`**: **100 family độc lập × 200 tasks = 20.000 tasks/phương pháp**, seeds 44–143. Mỗi task có 32 quan sát, chia raw train/validation 26/6 trước augmentation. Có tổng cộng 40.000 task losses của LFBO + GP.

| Thuật toán | Paper — Average Loss ↓ | Thực nghiệm của Phước — Weighted BCE ± SE ↓ | Paper — CVaR ↓ | Thực nghiệm của Phước — CVaR₀.₈ ± SE ↓ |
|---|---:|---:|---:|---:|
| LFBO | 0.5823 | 0.523973 ± 0.027310 | 0.9876 | 1.303971 ± 0.138380 |
| GP | 0.7891 | 0.328993 ± 0.001370 | 1.3245 | 0.600549 ± 0.002292 |

**`NOT_PAPER_EQUIVALENT`**: cột paper là số tham chiếu Table 1; cột thực nghiệm lấy từ [summary.json](results/published/lfbo-gp-full100-20261114/summary.json). ± là **sample SE trên 100 family metrics**, không phải độ lệch chuẩn hoặc SE trên 20.000 tasks. Đây là classifier loss, không phải normalized BO regret.

Phản hồi tác giả xác nhận Weighted BCE theo task và CVaR trên task losses. Tuy nhiên, seed list/protocol đầy đủ của Table 1 chưa được xác minh; held-out 26/6 và GP evaluator của nhóm là lựa chọn riêng. Vì vậy bảng chỉ đối chiếu mô tả, không kết luận phương pháp của nhóm vượt paper.

![Weighted BCE và CVaR của LFBO + GP](results/published/lfbo-gp-full100-20261114/BCE_CVaR_same_protocol.png)

## Chạy trên Ubuntu

Dùng **Python 3.12.14** và dependencies CPU trong `requirements.txt`. Không cần checkpoint tải ngoài hoặc dịch vụ theo dõi.

```bash
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 -B main.py --mode preflight

# Smoke thật: 3 tasks, RF 1.000 cây và GP đúng cấu hình.
python3 -B -u main.py --mode smoke --run-id lfbo-gp-smoke-user --workers 1

# Một family gồm 200 tasks.
python3 -B -u main.py --mode family --run-id lfbo-gp-family200-user --workers 1

# Chỉ chạy full khi chủ động dùng đủ hai cờ xác nhận.
# 100 family × 200 tasks = 20.000 fits/phương pháp.
python3 -B -u main.py --mode full --run-id lfbo-gp-full100-user --workers 2 \
  --confirm-full --acknowledge-independent-protocol

# Tiếp tục đúng run, dùng lại các unit đã hoàn thành.
python3 -B -u main.py --mode resume --run-id lfbo-gp-full100-user --workers 2 \
  --confirm-full --acknowledge-independent-protocol
```

Chọn run ID mới để giữ kết quả cũ. CLI không chạy thực nghiệm khi thiếu `--mode`; help/preflight không fit mô hình. Mặc định run mới lưu **hai checkpoint native: LFBO và GP của task đầu tiên, family đầu tiên**. Dùng `--checkpoints none` để tắt, hoặc `--checkpoints all` nếu muốn lưu mọi mô hình và đã chuẩn bị dung lượng. Policy được khóa trong `job.json`; resume giữ nguyên policy.

## Implementation và metric

- LFBO giữ implementation hiện có: EI augmentation, Random Forest 1.000 cây, gamma = 1/3. Ngưỡng tau và hệ số chuẩn hóa weights chỉ tính từ train.
- GP giữ implementation hiện có: exact Gaussian process, Constant × Matérn 5/2, latent posterior. Track B dùng `q = EI_tau / (EI_tau + U_train)`; đây là **`GROUP_DERIVED_EVALUATOR / NOT_AUTHOR_VERIFIED`**, chưa phải GP classifier evaluator được tác giả xác nhận.
- Task Weighted BCE = `sum(weight × BCE) / số augmented validation rows`; mẫu tốt có thể xuất hiện thêm ở positive rows. Không chia cho tổng weights.
- CVaR₀.₈ trong mỗi family dùng đúng `k=max(1,int(N*(1-alpha)))` task losses lớn nhất. Với N=200 và float Python, k=39. Bảng cuối lấy mean ± sample SE của 100 family means/CVaRs.

Cấu hình portable ở `configs/lfbo_gp_unified/protocol_v2.json`; tham số khoa học, seed và split giữ nguyên v1. Nguồn tác giả, upstream LFBO, licenses và provenance nằm trong `vendor/`, `NOTICE`, `LICENSE`.

## Bằng chứng kết quả

Kết quả tại [results/published/lfbo-gp-full100-20261114/](results/published/lfbo-gp-full100-20261114/), khoảng **29.5 MB** gồm cả dữ liệu NPZ gọn:

| File | Nội dung |
|---|---|
| `summary.json`, `run_metrics.json` | Kết quả tổng hợp và metrics của 100 family |
| `task_losses.csv` | Đủ 40.000 task losses LFBO/GP |
| `paired_task_differences.csv` | Chênh lệch loss theo từng task chung |
| `config.json`, `environment.json`, `job.json`, `registration.json` | Cấu hình gốc, runtime, seeds và đăng ký lần chạy |
| `logs/`, các biểu đồ PNG | Log chạy, kiểm tra archive gốc, đối chiếu PyTorch và biểu đồ |
| `evidence/meta_dataset.npz` | Đủ 640.000 raw observations của 20.000 tasks, train/validation indices và train-only constants |
| `evidence/classifier_predictions.npz` | Predictions/posterior, labels/weights, logits và loss của cả LFBO/GP trên validation |
| `evidence/evidence_manifest.json`, `publication_manifest.json` | SHA256, nguồn từng file và phạm vi bộ công bố |

Bộ công bố hỗ trợ kiểm tra từ raw split, weights và predictions → Weighted BCE từng task → CVaR từng family → bảng tổng hợp. Toàn bộ JSON archive/source snapshot khoảng 1,45 GB vẫn giữ local tại `results/lfbo_gp_unified/unified-b-full100-20261114/`. Bộ NPZ không chứa checkpoint mô hình full100 và không dùng để resume. Các báo cáo Markdown trung gian và ZIP giữ ngoài Git.

Bộ [smoke có checkpoint](results/published/lfbo-gp-evidence-smoke-20261118/) được chạy thật trên **3 tasks**, RF 1.000 cây và GP cùng cấu hình. Hai checkpoint đại diện cho task đầu tiên được lưu lúc fit; đây **không phải checkpoint full100**. Dự đoán của cả 3 tasks khớp chính xác các tasks tương ứng trong run full100 cũ.

Các run mới tiếp tục lưu raw archive riêng; có thể xuất thêm bộ NPZ gọn gồm **toàn bộ raw X/y, split train/validation, labels/weights, probabilities, logits, task losses**, kèm checkpoint đại diện nếu đã lưu khi fit. Không tạo checkpoint hồi tố bằng cách fit lại. Run full100 cũ chưa lưu checkpoint mô hình nên bản xuất sẽ ghi `NOT_SAVED_DURING_ORIGINAL_RUN`.

## Kiểm chứng và replay

Xem/kiểm tra một raw archive đầy đủ đã có trên máy, không training/inference/objective:

```bash
python3 -S -B main.py --mode show --run-id lfbo-gp-full100-user
python3 -S -B main.py --mode compare --run-id lfbo-gp-full100-user
python3 -S -B main.py --mode verify --run-id lfbo-gp-full100-user
```

Kiểm tra ngay bộ full100 có sẵn sau khi clone; không cần raw JSON archive:

```bash
python3 -B main.py --mode verify-evidence \
  --evidence-dir results/published/lfbo-gp-full100-20261114/evidence

# Replay hai checkpoint đại diện từ smoke đã công bố (chỉ inference).
python3 -B main.py --mode replay-checkpoints \
  --evidence-dir results/published/lfbo-gp-evidence-smoke-20261118
```

Xuất và kiểm tra bằng chứng từ smoke vừa chạy; destination phải mới:

```bash
python3 -B main.py --mode export-evidence --run-id lfbo-gp-smoke-user \
  --evidence-dir results/evidence/lfbo-gp-smoke-user
python3 -B main.py --mode verify-evidence \
  --evidence-dir results/evidence/lfbo-gp-smoke-user
python3 -B main.py --mode replay-checkpoints \
  --evidence-dir results/evidence/lfbo-gp-smoke-user
```

- `export-evidence`: đọc và kiểm tra raw archive rồi xuất `meta_dataset.npz`, `classifier_predictions.npz`, metadata, SHA256 và checkpoint đại diện đã có; không fit hoặc sinh objective data mới.
- `verify-evidence`: dùng arrays đã lưu để kiểm tra split, train-only weights, xác suất, logits, Weighted BCE, task CVaR và SE; không import mô hình, không inference và không ghi đè file.
- `replay-checkpoints`: load estimator native do pipeline lưu, chạy **inference** để đối chiếu predictions/posterior; không training, không tối ưu lại GP và không tính objective. Thiếu checkpoint báo lỗi, không tự fit bù. Replay dùng đúng runtime đã khóa.

Bộ NPZ là bằng chứng dữ liệu, không phải archive đầy đủ của mọi mô hình/source snapshot và không dùng để resume. Các lệnh báo lỗi khi thiếu dữ liệu. Archive v1 có thể xem bằng snapshot gốc; resume v1 phải dùng code/runtime v1, không sửa protocol lock để vượt kiểm tra.

## Cấu trúc và kiểm thử

```text
main.py                        Entry point Ubuntu
models/lfbo.py, models/gp.py    Implementation LFBO/GP
experiments/lfbo_gp_unified/    Chạy, resume, xuất bằng chứng, verify/replay
configs/lfbo_gp_unified/        Protocol, nguồn và split đã đăng ký
tests/                         Unit/integration/reproducibility checks
vendor/                        Nguồn tham chiếu và licenses
results/published/              Kết quả được đưa lên Git
results/lfbo_gp_unified/        Raw archive local từng run
results/evidence/               Bộ NPZ local và checkpoint đại diện
```

```bash
.venv/bin/python -B -m pytest
```

Kiểm tra lần cập nhật bằng chứng: **116 tests PASS** (110 tests chính và 6 đối chiếu PyTorch); smoke thật và replay checkpoint PASS. Kết quả test lưu trong `results/published/lfbo-gp-evidence-smoke-20261118/logs/`. Tests dùng fixture nhỏ; không tự chạy benchmark full100. Đối chiếu PyTorch/source AST tùy chọn trong môi trường CPU Python 3.11 riêng:

```bash
python3.11 -m venv .venv-torch
.venv-torch/bin/python -m pip install -r requirements-torch-check.txt
.venv-torch/bin/python -B -m pytest tests/test_lfbo_gp_unified_torch.py
```

# MALIBO và PFN trên Forrester

## Chạy thực nghiệm

Dùng môi trường `.venv` hiện có:

```powershell
.venv/Scripts/python main.py
```

Mặc định: **200 task, 32 mẫu/task, 100 epoch, batch 10, seed 44,
1 target và tổng 16 đánh giá BO** (2 điểm khởi tạo + 14 đề xuất).
Kết quả được ghi vào `results/author_reply/`.

```powershell
# Kiểm thử
.venv/Scripts/python -W error::RuntimeWarning -m pytest -q
# Chạy nhanh để kiểm tra pipeline
.venv/Scripts/python main.py --quick
# Chạy nhiều target và lưu riêng
.venv/Scripts/python main.py --runs 100 --output results/multiple_targets
# Kiểm chứng từ checkpoint, dữ liệu và log đã lưu
.venv/Scripts/python scripts/verify_results.py results/author_reply
# Tạo lại báo cáo sau khi kiểm chứng
.venv/Scripts/python scripts/write_run_report.py results/author_reply
```

Nếu tạo môi trường mới:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Cấu trúc

```text
main.py                 Entry point
experiment.py           Cấu hình, huấn luyện, BO và xuất kết quả
metrics.py              Weighted BCE, CVaR và normalized regret
benchmarks/             Họ hàm Forrester và dữ liệu task
models/                 Adapter MALIBO, PFN và audit classifier
vendor/                 Implementation công khai của tác giả, license, provenance
scripts/                Kiểm chứng checkpoint và tạo báo cáo
tests/                  Kiểm thử hai phương pháp và protocol
results/author_reply/   Kết quả đầy đủ đã kiểm chứng
16738-AAAI26.LiH-RU.pdf Bài báo gốc
```

Giữ `.venv/` để chạy ngay; `.git/` chứa lịch sử phiên bản.

## Implementation và phép đo

MALIBO dùng [kho tác giả](https://github.com/Allen0497/MetaGameBO), commit
`489213ddcfbfe275a8b443c0d8b136ef6c36ad2c`, với DRO và k-DPP tắt.
Giữ nguyên thuật toán upstream; các bản vá tương thích và audit được ghi ở
`vendor/metagamebo_author/LOCAL_CHANGES.md`. Không xóa `vendor/` khi chạy.
Batch 10 của MALIBO là **10 dòng phân loại** theo trainer gốc;
PFN dùng **10 task/batch**.

- Task BCE = `sum(weight * BCEWithLogits) / number_of_classifier_rows`.
  LFBO nhân đôi các mẫu tốt, nên số dòng phân loại có thể lớn hơn 32.
- Macro BCE trung bình đều các task; micro BCE chia tổng loss cho tổng số dòng.
- CVaR dùng `alpha=0.8`, đúng `k=max(1,int(number_of_tasks*(1-alpha)))`.
  Theo float/int của Python, với 200 task thì `k=39`.
- Objective có regularization lưu riêng; không đưa regularization vào CVaR BCE.
- Normalized BO regret dùng cực trị trên lưới 20.001 điểm của từng target,
  báo cáo riêng khỏi BCE. Ngân sách 16 tính cả các điểm khởi tạo.

Classifier giữ split 80/20 theo dòng của tác giả, có thể chứa cùng task và
tọa độ LFBO ở train/validation. Audit chạy eval mode và giữ RNG huấn luyện.
Benchmark dùng cùng phân phối Forrester, với các luồng RNG source/target riêng,
paired random initial design và pool acquisition 512 điểm.

PFN huấn luyện cục bộ trên prior Forrester, không phải checkpoint pretrained
của bài báo. Query chỉ attention vào context; context length biến thiên 1–31;
head Normal dự đoán mean/std, acquisition dùng EI cho minimization.
Gaussian NLL chỉ lưu như diagnostic riêng, không đưa vào bảng BCE.

## Kết quả cần đọc

- `REPORT.md`: báo cáo và giới hạn diễn giải.
- `classifier_bce.csv`, `classifier_task_losses.csv`: BCE và CVaR theo task.
- `classifier_objective.csv`: objective có regularization.
- `bo_regret.csv`, `convergence.png`: kết quả BO.
- `bo_traces.json`, `curves.npz`: dữ liệu đầy đủ của vòng BO.
- `*_checkpoint.pt`, `meta_dataset.npz`, `classifier_predictions.npz`:
  checkpoint, dữ liệu và logits/labels/weights để kiểm chứng.
- `*_training.json`, `config.json`, `verification.json`, `pytest.txt`,
  `requirements-lock.txt`: lịch sử huấn luyện, cấu hình và kiểm tra.

Tác giả xác nhận không còn seed list và log gốc để bảo đảm khớp từng chữ số
Table 1. Một target/seed chưa đủ để kết luận ưu thế thống kê giữa hai phương pháp.

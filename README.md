# MetaGameBO_review

Phần thực nghiệm **LFBO** và **PFN** trong đồ án tổng quan MetaGameBO của môn Nhập môn Học máy.
Mục tiêu là chạy hai thuật toán trên benchmark Forrester, tổng hợp kết quả và biểu đồ để đưa vào slide báo cáo của nhóm.

## Hiện trạng

Đã có benchmark Forrester, bộ đánh giá chung, Random Search và **LFBO-EI dùng Random Forest**.
LFBO là baseline không meta-learning, chỉ dùng các quan sát target đã có.
PFN dùng checkpoint **PFNs4BO HEBO+ pretrained**, chạy inference trên CPU, không huấn luyện lại.
Các cấu hình hiện tại là thực nghiệm thăm dò; chưa xác nhận tương đương Table 1 của bài báo.

## Chuẩn bị môi trường

Sử dụng Python 3.11 hoặc 3.12. Chạy tại thư mục gốc repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Chạy kiểm tra nhanh trên CPU

```bash
pytest -q
python -m experiments.run_forrester --config configs/forrester_smoke.yaml --model random --mode smoke --wandb-mode disabled
python -m experiments.run_forrester --config configs/lfbo_smoke.yaml --model lfbo --mode smoke --wandb-mode disabled
```

Cấu hình kiểm tra gồm 3 target task × 3 seed, mỗi run có tổng ngân sách 8 lần đánh giá, bao gồm 2 điểm khởi tạo.
Kết quả được lưu trong `results/`, gồm CSV, JSON và biểu đồ. Thêm `--resume` khi chạy lại cùng cấu hình để tiếp tục các run còn thiếu.

LFBO smoke dùng RF 128 cây, 1024 ứng viên và gamma=1/3. W&B disabled, không có URL công khai.
Trọng số EI và fallback được kiểm tra trong `tests/test_lfbo*.py`.

## Phần thực nghiệm tiếp theo

So sánh paired LFBO, PFN và Random qua các cấu hình development
(3 task × 10 seed × 16 đánh giá). Development và cấu hình 100 lần lặp **chưa chạy**.

Cấu hình chạy chính thức và cách đối chiếu chỉ số với bài báo sẽ được xác nhận trước khi thực nghiệm.

## PFN: môi trường riêng và smoke

PFNs4BO cũ cần sklearn<1.2, nên giữ `.venv` LFBO và dùng `.venv-pfn` với Python 3.11:

```bash
bash scripts/setup_pfn_env.sh
.venv-pfn/bin/python -m pytest -q -m pfn_real
.venv-pfn/bin/python -m experiments.run_forrester --config configs/pfn_tiny.yaml --model pfn --mode smoke --wandb-mode disabled
.venv-pfn/bin/python -m experiments.run_forrester --config configs/pfn_smoke.yaml --model pfn --mode smoke --wandb-mode disabled
```

Tiny có 1 task × 1 seed × 4 đánh giá; smoke có 3 × 3 × 8. Checkpoint được tải
tường minh từ nguồn chính thức và kiểm tra checksum, nằm trong `.cache/`, không commit weights.
PFN nhận `-loss`, X normalize theo bounds; EI dùng context standardization từ observed y.
Manifest smoke dùng chung với LFBO/Random, lưu trong `configs/manifests/forrester_shared_smoke.json`.
Standard pytest dùng unit doubles và không tải weights; `-m pfn_real` chạy checkpoint thật.

Tham khảo: [LFBO gốc](https://github.com/lfbo-ml/lfbo), [MetaGameBO](https://github.com/Allen0497/MetaGameBO), [yêu cầu đồ án](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/).

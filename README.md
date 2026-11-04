# MetaGameBO_review

Phần thực nghiệm **LFBO** và **PFN** trong đồ án tổng quan MetaGameBO của môn Nhập môn Học máy.
Mục tiêu là chạy hai thuật toán trên benchmark Forrester, tổng hợp kết quả và biểu đồ để đưa vào slide báo cáo của nhóm.

## Hiện trạng

Đã có benchmark Forrester, bộ đánh giá chung và Random Search để kiểm tra pipeline.
**LFBO và PFN chưa được triển khai hoặc chạy thực nghiệm.** Kết quả kiểm tra pipeline hiện tại chưa phải kết quả tái lập bài báo.

## Chuẩn bị môi trường

Sử dụng Python 3.11 hoặc 3.12. Chạy tại thư mục gốc repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Kiểm tra pipeline

```bash
pytest -q
python -m experiments.run_forrester --config configs/forrester_smoke.yaml --model random --mode smoke --wandb-mode disabled
```

Cấu hình kiểm tra gồm 3 target task × 3 seed, mỗi run có tổng ngân sách 8 lần đánh giá, bao gồm 2 điểm khởi tạo.
Kết quả được lưu trong `results/`, gồm CSV, JSON và biểu đồ. Thêm `--resume` khi chạy lại cùng cấu hình để tiếp tục các run còn thiếu.

## Phần thực nghiệm tiếp theo

- Tích hợp LFBO và PFN vào bộ đánh giá chung.
- Chạy hai thuật toán với cùng task, seed, điểm khởi tạo và ngân sách đánh giá.
- Tổng hợp kết quả qua các seed, vẽ biểu đồ và phân tích để chuẩn bị slide.

Cấu hình chạy chính thức và cách đối chiếu chỉ số với bài báo sẽ được xác nhận trước khi thực nghiệm.

Tham khảo: [mã nguồn tác giả](https://github.com/Allen0497/MetaGameBO), [yêu cầu đồ án](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/).

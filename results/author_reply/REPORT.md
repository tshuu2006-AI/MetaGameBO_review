# Kết quả tái lập theo phản hồi tác giả

Ngày báo cáo: 13/11/2026 (Asia/Bangkok).

Cấu hình đầy đủ: 200 source task, 32 mẫu/task, 100 epoch, batch 10, seed 44, 1 target, tổng 16 điểm objective/target/phương pháp (gồm 2 điểm khởi tạo). Thiết bị: cpu.

MALIBO dùng implementation của Allen0497/MetaGameBO, commit `489213ddcfbfe275a8b443c0d8b136ef6c36ad2c`, tắt cả DRO và k-DPP. PFN là mô hình huấn luyện cục bộ trên prior Forrester; không phải checkpoint pretrained của bài báo.

## Weighted BCE của MALIBO

| Split | Task | Dòng phân loại | Macro BCE | Micro BCE | CVaR BCE | k |
|---|---:|---:|---:|---:|---:|---:|
| train | 200 | 6881 | 0.330755450 | 0.331522650 | 0.394570606 | 39 |
| validation | 200 | 1719 | 0.561664654 | 0.572715291 | 1.138041167 | 39 |

Task loss = tổng weighted BCE chia số dòng phân loại của task. CVaR dùng alpha 0.8 và đúng `max(1,int(number_of_tasks*(1-alpha)))`; không dùng ngưỡng phân vị, KDE hoặc objective có regularization.

Split 80/20 theo dòng đúng trainer công khai. Với 8.600 dòng và float Python, cấu hình đầy đủ chia 6.881/1.719 dòng. Validation có cùng source task id và có thể trùng tọa độ LFBO với train. Không gọi đây là BCE trên task chưa thấy.

## Objective được lưu riêng

| Split | Regularization | Micro BCE + regularization |
|---|---:|---:|
| train | 0.018303979 | 0.349826629 |
| validation | 0.018303979 | 0.591019270 |

Regularization giữ MetaLoss gốc: 0.1 × tổ hợp KS/covariance của task embedding. Audit chạy eval mode, không đưa embedding noise vào loss, không tiêu thụ RNG huấn luyện.

## Normalized BO regret

| Phương pháp | Regret cuối | SE giữa các target | CVaR regret |
|---|---:|---:|---:|
| MALIBO | 0.0000279061 | 0.0000000000 | 0.0000279061 |
| PFN | 0.0012788335 | 0.0000000000 | 0.0012788335 |

Regret và CVaR regret nằm trong bảng riêng; không tương đương BCE/CVaR của Table 1. Đường hội tụ: `convergence.png`; toàn bộ X/y: `bo_traces.json`.

## PFN và kiểm chứng

Attention chặn mọi cột query, nên context không đọc query và mỗi query chỉ đọc context. Không đưa query label vào input. Head Normal dự đoán cả mean và std dương, acquisition là EI giải tích cho minimization. Inference không cập nhật trọng số.

Context length đã xuất hiện: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]. Mỗi epoch dùng đủ 200 task.

NLL train chuẩn hóa y theo context: epoch đầu 3.273426354; epoch cuối -0.147963432. Đây là diagnostic train; NLL có thể âm vì density liên tục có thể lớn hơn 1. Không đưa NLL/MSE vào bảng BCE.

Bộ kiểm thử: 18 passed, gồm mask/query independence, context biến thiên, predictive density hữu hạn, EI đối chiếu Monte Carlo, inference giữ nguyên trọng số, đúng số đánh giá BO, macro/micro BCE và CVaR floor rule.

Kiểm chứng độc lập (`verification.json`): checkpoint tái tạo khớp BCE trên từng split; mọi X/y được đối chiếu objective target; curve tái tính khớp; vendor SHA256 khớp. Dữ liệu, logits/labels/weights/task id và checkpoint được lưu để kiểm tra lại.

```powershell
.venv/Scripts/python scripts/verify_results.py results/author_reply
```

## Giới hạn diễn giải

Tác giả không còn seed list và toàn bộ log gốc, nên không bảo đảm khớp từng chữ số Table 1. Benchmark giữ phân phối tham số gốc nhưng dùng các luồng RNG tách biệt của dự án, paired random initial design và pool acquisition 512 điểm. PFN chưa có quy tắc BCE được tác giả xác nhận và chưa dùng checkpoint pretrained gốc.

Chạy một target/seed cho SE bằng 0 theo quy ước bảng, không phải bằng chứng phương sai bằng 0 hay ưu thế thống kê. Trên target mặc định này, điểm khởi tạo đã có regret khoảng 0.001279; PFN không cải thiện incumbent sau khởi tạo. Cần nhiều target/seed trước khi kết luận so sánh tổng quát.

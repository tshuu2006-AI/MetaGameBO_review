# Audit archive hiện tại và mức độ khớp với paper

## 1. Kết luận

Archive `MALIBO.rar` là prototype tự viết theo một số ý tưởng mô tả trong paper MetaGameBO, **không phải source code chính thức của MALIBO (Pan et al., 2024)**.

Các dấu hiệu trực tiếp:

- `scripts/meta_train.py`, `scripts/run_benchmark.py` và `models/residual_booster.py` rỗng.
- `main.py` dùng `MetaLearnerNet` MLP concat `(x,z)` và BCE trên nhãn quantile; đây không phải `MetaBLOR` trong implementation MALIBO.
- `core/malibo.py` tối ưu latent `z` bằng BCE + prior L2, dùng Hessian/Laplace, rồi GradientBoostingRegressor trên residual. Official MALIBO dùng task-agnostic meta-model + Bayesian logistic regression layer, Thompson sampling của classifier và `GradientBoostingClassifier` được khởi tạo từ classifier meta-learned.
- `requirements.txt` trong archive có dòng `git+https://github.com` không hoàn chỉnh, không cài được một repo cụ thể.

## 2. MALIBO gốc cần có gì?

Theo implementation chính thức:

1. Meta-training tạo classifier dùng chung từ nhiều meta-task (`meta_fit`).
2. Mỗi observation được chuyển thành nhãn tốt/xấu dựa trên quantile `gamma`.
3. Sample weights dựa trên utility EI: `max(tau - y, 0)`, sau đó cân bằng lớp.
4. Trên target task, cập nhật Bayesian logistic regression; Thompson sampling để chọn candidate.
5. Khi đủ điểm thuộc cả hai lớp, fit `GradientBoostingClassifier(init=meta_classifier, ...)` để hiệu chỉnh acquisition.

Không nên dùng `GradientBoostingRegressor` residual và gọi đó là MALIBO baseline.

## 3. PFN/PFN s4BO

Bài MetaGameBO ghi PFN (không phải PNF), trích Müller et al. (2023). PFNs4BO sử dụng Prior-data Fitted Network làm surrogate dự đoán posterior predictive distribution qua in-context learning. Package cung cấp model pretrained và `TransformerBOMethod`; mọi feature và pending candidate đưa vào interface phải chuẩn hóa về [0,1].

## 4. Sai lệch cần sửa so với archive

| Thành phần | Archive hiện tại | Baseline workspace |
|---|---|---|
| MALIBO | MLP tự xây + latent z tối ưu thủ công | Gọi class MALIBO upstream |
| Meta training | z=0 cố định; train loop nằm trong main.py | MetaBLOR `meta_fit` trên task data |
| Target adaptation | MAP của latent z + Hessian inverse | Bayesian logistic regression layer của MALIBO |
| Auxiliary model | GradientBoostingRegressor trên residual | GradientBoostingClassifier khởi tạo từ MALIBO classifier |
| PFN | Chưa có | PFNs4BO `TransformerBOMethod` pretrained |
| Benchmark | Forrester dịch và scale tự đặt | Forrester task family trong source MALIBO |
| Thống kê | CVaR tham số alpha mơ hồ; không có SE | CVaR95 upper-tail 5% + standard error |

## 5. Các điểm paper chưa công bố trong main paper

Để tuyên bố tái lập chính xác Table 1 cần tìm thêm Appendix/supplementary:

- cụ thể hóa task distribution của Forrester và cách lấy target/outlier tasks;
- số initial observations và tổng evaluation budget;
- model PFN đã dùng (HEBO+ hay BNN) và các tùy chọn inference;
- noise, candidate pool và mọi random seed;
- cách xác định `f_min/f_max` cho normalized regret.

Workspace hiện công khai giả định để chạy baseline có thể lặp lại, không tự gán các giả định này thành chi tiết của paper.

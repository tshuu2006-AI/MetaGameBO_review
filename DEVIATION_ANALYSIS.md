# Deviation analysis: MetaGameBO (AAAI-26) so với dự án tái tạo

Tài liệu này trả lời hai câu hỏi:

1. Dự án hiện tại đã khớp với bài báo chưa (mục 2)?
2. Phần đánh giá của bài báo và của dự án khác nhau ở đâu, và vì sao (mục 3–5)?

Quy ước độ chắc chắn dùng trong tài liệu:

- **[đã đo]**: tôi chạy và đo trực tiếp trong dự án; lệnh tái lập ở mục 8.
- **[đọc code]**: suy ra từ việc đọc code, có ghi file và dòng.
- **[quan sát hình]**: đọc bằng mắt trên hình của bài báo, chưa số hoá.
- **[không rõ]**: bài báo không nêu, không kiểm chứng được.

---

## 1. Nguồn đối chiếu

| Nguồn | Ghi chú |
|---|---|
| `paper.pdf` | Bản AAAI-26, 9 trang, **không có appendix**. Bài báo nhiều lần ghi "chi tiết ở Appendix" (siêu tham số, mô tả benchmark, ablation đầy đủ). Tôi không tìm thấy bản arXiv có appendix. |
| Code tác giả | `models/metagamebo_original/author/` (github.com/Allen0497/MetaGameBO). Đây là bản sửa từ code MALIBO của Bosch. |
| Bài báo MALIBO | arXiv 2307.03565v3. MetaGameBO ghi "following the evaluation protocol of MALIBO". |
| Code MALIBO | github.com/boschresearch/meta-learning-likelihood-free-bayesian-optimization. |
| Kết quả của dự án | Forrester: 10 seed, đủ 7 phương pháp (máy local). HPOBench: 5 seed × 4 task, mới có 5/9 phương pháp (local). NAS-Bench-201: 5 seed × 3 task, đủ 9 phương pháp (Kaggle, tôi chỉ có hình bạn gửi). HPO-B bản continuous: ABLR, MALIBO, PFN đủ 430 lần chạy; original mới 3/16 search space (local). |

---

## 2. Dự án đã khớp bài báo chưa

Kết luận ngắn: phần bài toán, dữ liệu và thuật toán remake khớp ở mức bài báo mô tả được. Có **sáu chỗ chưa khớp** (mục 2.3), trong đó chỗ ảnh hưởng nhiều nhất là giao thức HPO-B.

### 2.1 Thuật toán: bài báo, remake và code tác giả

| Thành phần | Bài báo | Remake (`models/metagamebo_remake`) | Code tác giả |
|---|---|---|---|
| Ngưỡng CVaR (Eq. 1–2) | Ước lượng ξ bằng KDE Gauss, `T_hard = {L_i ≥ ξ}` | Khớp (`game.py`, `kde_quantile`) | **Không dùng KDE**: sắp xếp và lấy `max(1, int(M·0.2))` task loss lớn nhất (`trainer.py:317–320`). Hàm KDE có định nghĩa nhưng không được gọi. |
| Nhãn và trọng số (Eq. 3) | Lớp tham chiếu là các điểm `y ≥ y_th` | **Mặc định không khớp**: dùng cách của LFBO (lớp 0 = mọi điểm, điểm tốt mang cả hai nhãn). Có tuỳ chọn `weighting="paper"`. | Cách của LFBO (`MetaGameBO.py:236–237`), giống code MALIBO. |
| k-DPP, `K = diag(w)` | Lấy mẫu tập con mỗi vòng lặp | Khớp, lấy mẫu chính xác (`kdpp.py`, `DiagonalKDPP`) | **Không lấy mẫu khi huấn luyện.** `use_kdpp` chỉ được dùng để đặt tên file trong `trainer.py`. Tác dụng duy nhất là `_apply_kdpp_weights`: sửa trọng số một lần trước khi huấn luyện. |
| Cập nhật trọng số (Eq. 6) | `w ← w·exp(η_s (ℓ − ℓ̄)/q)` theo loss của mô hình | Khớp. Thêm kẹp số mũ, chuẩn hoá trung bình 1 theo task (không có trong bài báo, để ổn định số). | "Loss" đưa vào là `1 − nhãn`, không phải loss của mô hình (`MetaGameBO.py:286–288`, nhánh dự phòng khi độ dài lệch). |
| Regulariser | `β·KL[q_emp(z) ‖ p(z)]` | KS + hiệp phương sai của MALIBO, hệ số 0.1 | Giống remake (kế thừa MALIBO). |
| Bước gradient | Một bước mỗi vòng trên `T_hard` | Khớp: 2000 bước Adam full-batch | Mỗi epoch: một bước cho **từng mảnh ~1 mẫu** của các task được chọn; batch 10; LR giảm 0.999/epoch. |
| Laplace (Eq. 4) | `Σ_N` = nghịch đảo Hessian | Khớp (`hessian="exact"`) | Số hạng Hessian của code MALIBO (khác công thức). |
| Thompson sampling, GB residual, điểm đầu tiên | Như MALIBO | Khớp | Khớp |
| Siêu tham số | Không nêu trong 9 trang | `α_task=0.8`, `k=0.4n`, `η_s=0.1` (lấy từ mặc định code tác giả) | Như cột bên |

Hai hệ quả cần nhớ khi đọc kết quả:

- **"Original" và "remake" là hai thuật toán khác nhau.** Code tác giả không cài mức sample của Algorithm 1 (dòng 7–9); remake thì cài đúng theo bài báo.
- **MALIBO và remake khác nhau cả ở Hessian** (`"upstream"` so với `"exact"`), không chỉ ở phần game. Khi so regret BO giữa hai phương pháp này, khác biệt có thể đến từ Hessian.

### 2.2 Giao thức thí nghiệm

| Hạng mục | Bài báo | Dự án | Khớp? |
|---|---|---|---|
| Forrester: họ hàm | `a∈[0.2,3]`, `b∈[−5,15]`, `c∈[−5,5]` (theo code) | Giống | Có |
| Forrester: cấu hình | Không nêu; email tác giả: 200 task × 32 điểm, 100 epoch, batch 10 | Giống | Có |
| Table 1: thước đo | Văn bản ghi "normalized regret"; email tác giả: loss phân loại (BCE có trọng số) | Loss phân loại, theo email | Có (theo tác giả) |
| Số lần chạy | "100 independent runs" | Forrester 10 seed; AutoML 5 seed | **Không** |
| Ablation trên Forrester | Có (task-only, sample-only) | Trước đây chưa có; nay có hai script (mục 8) | Đã bổ sung |
| HPOBench | 4 dataset UCI, 9 chiều | FCNet, 62.208 cấu hình, 4 dataset | Có |
| NAS-Bench-201 | 15.625 kiến trúc, 3 dataset | Giống; mã hoá 6 chiều thứ tự | Có |
| Meta-data | Leave-one-task-out, 512 điểm mỗi task nguồn | Giống | Có |
| HPO-B | Split meta-train/test có sẵn, 5 điểm khởi tạo cố định | Split và 5 điểm khởi tạo giống; **bản continuous (surrogate)**, meta-train lấy 512 điểm/dataset | **Không** (xem 2.3) |
| Số bước | 500 trên cả ba hình | 500 | Có, trừ HPO-B (xem 5.2) |
| Regret chuẩn hoá | `(f − f_min)/(f_max − f_min)` trên mọi đánh giá có sẵn của task | Giống với bảng tra; HPO-B continuous dùng thống kê của surrogate | Có, trừ HPO-B |
| Tổng hợp | Trung bình ± sai số chuẩn | Giống; thêm trung vị và trung bình hình học | Có |
| Baseline | Random, GP, LFBO, ABLR, RGPE, MetaBO, PFN, MALIBO | Thiếu MetaBO; tất cả là bản tự cài hoặc tự bọc | Một phần |

### 2.3 Sáu chỗ chưa khớp

| # | Chỗ chưa khớp | Mức ảnh hưởng | Trạng thái |
|---|---|---|---|
| 1 | **HPO-B chạy bản continuous, 500 bước.** Code tác giả và code MALIBO chỉ có bản discrete (`benchmarks/hpob.py:184` ném `NotImplementedError` khi `continuous=True`). Hình của MALIBO chỉ có 100 bước. | Cao: toàn bộ ô HPO-B không so được với bài báo | **Đã thêm `--discrete`** (kết quả lưu riêng `hpob_discrete`). Chưa chạy đầy đủ. |
| 2 | **MALIBO baseline không có early stopping.** Code MALIBO gốc dừng sớm theo val loss (patience 20 epoch) và khôi phục trọng số tốt nhất. | Trung bình: Table 1 hàng MALIBO là 0.519 thay vì khoảng 0.43 | Chưa sửa (sẽ phải chạy lại MALIBO ở mọi benchmark) |
| 3 | **Remake dùng nhãn LFBO**, không phải Eq. 3 như bài báo viết. | Cao với Table 1: thang đo loss đổi 4 lần (0.53 → 0.13) | Lựa chọn có chủ ý, để mọi phương pháp chung một thước đo. Có sẵn `weighting="paper"`. |
| 4 | **Số seed** ít hơn bài báo 10–20 lần. | Trung bình: đường regret gồ ghề, chạm đúng 0 | Chạy thêm bằng `--seeds` |
| 5 | **Baseline tự cài; thiếu MetaBO.** | Cao với Figure 3 (xem 5.2) | Không thể khớp hoàn toàn |
| 6 | **Ứng viên trên bảng tra:** mỗi bước chỉ chấm 5120 điểm chưa đánh giá, chọn ngẫu nhiên. Code tác giả (HPO-B) chấm mọi điểm còn lại. | Thấp đến trung bình với bảng lớn (đến 102.306 điểm) | Chưa sửa (RGPE và PFN quá chậm nếu chấm hết) |

Hai điểm nhỏ khác:

- "Average Loss" của hàng original lấy từ `Final Val Loss` của trainer tác giả. Giá trị này tính ở chế độ train (có nhiễu embedding) và cộng thêm 0.1 × regulariser. Chênh lệch so với cách tính của các hàng khác dưới 0.02 [đã đo].
- Số bước của dự án tính cả 5 điểm khởi tạo; MALIBO chỉ vẽ phần sau khởi tạo.

---

## 3. Table 1 (Forrester)

### 3.1 Số liệu

Loss phân loại trên mẫu held-out, trung bình ± sai số chuẩn. Dự án: 10 seed.

| Phương pháp | Bài báo: Avg | Bài báo: CVaR | Dự án: Avg | Dự án: CVaR |
|---|---|---|---|---|
| Random | 0.8945 ± 0.00156 | 1.4532 ± 0.00892 | 0.8325 ± 0.0138 | 1.6132 ± 0.0347 |
| GP | 0.7891 ± 0.00134 | 1.3245 ± 0.00756 | 2.4916 ± 0.0291 | 4.6918 ± 0.0545 |
| LFBO | 0.5823 ± 0.00089 | 0.9876 ± 0.00445 | 0.5894 ± 0.0179 | 1.6356 ± 0.0819 |
| PFN | 0.5247 ± 0.00076 | 0.8934 ± 0.00523 | 0.9441 ± 0.0137 | 2.1989 ± 0.0351 |
| MALIBO | 0.4277 ± 0.00048 | 0.8195 ± 0.00376 | 0.5189 ± 0.0102 | 0.9985 ± 0.0377 |
| MetaGameBO (original) | 0.0183 ± 0.00003 | 0.0934 ± 0.00074 | 0.0173 ± 0.0017 | 0.0843 ± 0.0062 |
| MetaGameBO (remake) | (cùng hàng trên) | | 0.6440 ± 0.0086 | 0.9975 ± 0.0213 |

### 3.2 Các lệch và lý do

**T1. Original khớp bài báo, remake thì không (0.017 so với 0.644).** [đã đo]

Lý do: trong code tác giả, con số 0.018 đến từ việc **đổi trọng số của chính thước đo**, không phải từ mô hình tốt hơn.

- `_apply_kdpp_weights` (`MetaGameBO.py:257–321`) sửa trọng số mẫu **trước** khi chia train/validation. Val loss sau đó được tính bằng trọng số đã sửa.
- Đối tượng k-DPP được dùng lại giữa các task có cùng số dòng. Trên Forrester mọi task có 43 dòng, nên trọng số tích luỹ qua 200 task. Tỉ trọng của lớp "tốt" giảm từ 50% xuống **0.7%**.
- Khi lớp tốt gần như không còn trọng số, mô hình chỉ cần đoán "xấu" ở mọi nơi là loss gần 0. Bộ phân loại huấn luyện ra gán P(tốt) ≈ 0.01 cho các điểm tốt.
- Đo lại trên **cùng các mẫu held-out** bằng trọng số Eq. 3 gốc: bản đầy đủ cho **5.41 / 13.59**, trong khi MALIBO thật cho 0.52 / 1.00.
- Remake cài mức sample đúng theo Eq. 6 nên không có hiệu ứng này.

Bảng dưới là logic của `run_forrester.py`, seed 44 và 45, CPU (`results/forrester_author_ablation.md`):

| Cờ của script | k-DPP thực tế | Tỉ trọng lớp tốt | Val Loss / CVaR do trainer báo | Bài báo | Cùng mẫu, trọng số Eq. 3 gốc |
|---|---|---|---|---|---|
| không cờ (hàng MALIBO) | bật, k = 0.8n | 0.161 | 0.399 / 0.726 | 0.4277 / 0.8195 | 0.769 / 1.723 |
| `--use_dro` | bật, k = 0.8n | 0.161 | 0.333 / 0.595 | 0.335 / 0.589 ¹ | 0.653 / 1.422 |
| `--use_kdpp` | bật, k = 0.4n | 0.007 | 0.018 / 0.094 | 0.121 / 0.242 ¹ | 4.740 / 10.822 |
| `--use_dro --use_kdpp` | bật, k = 0.4n | 0.007 | 0.017 / 0.090 | 0.0183 / 0.0934 | 5.414 / 13.588 |
| không cờ, tắt hẳn k-DPP | tắt | 0.500 | 0.549 / 0.999 | – | 0.521 / 0.999 |
| `--use_dro`, tắt hẳn k-DPP | tắt | 0.500 | 0.558 / 0.877 | – | 0.560 / 0.876 |

¹ Suy ra từ phần trăm trong mục Ablation của bài báo (−21.7% / −28.1% và −71.8% / −70.5% so với MALIBO).

**T2. Hàng MALIBO: bài báo 0.428 / 0.820, dự án 0.519 / 0.999.** [đã đo] Có hai nguyên nhân, cùng chiều:

- **Script của tác giả vẫn bật k-DPP khi không truyền `--use_kdpp`.** Constructor có mặc định `use_kdpp=True, alpha=0.8`, và `run_forrester.py` chỉ ghi đè khi có cờ. Chạy không cờ cho 0.399 / 0.726, gần số của bài báo. Tắt hẳn k-DPP cho 0.549 / 0.999, gần số của dự án.
- **Không có early stopping.** Val loss của MALIBO trong dự án thấp nhất ở khoảng 10% quá trình huấn luyện (0.43 / 0.70) rồi tăng dần do overfit.

**T3. Hai hàng của Table 1 trong bài báo được đo bằng trọng số khác nhau.** [đã đo] Hàng MALIBO ứng với tỉ trọng lớp tốt 16% (hoặc 50% nếu là MALIBO thật), hàng MetaGameBO ứng với 0.7%. Vì thế mức "giảm 95.7%" so sánh hai thước đo khác nhau. Để thấy thang đo nhạy đến đâu: chỉ đổi cách gán nhãn Eq. 3 đã đưa MALIBO từ 0.528 xuống 0.128 (mục 4).

**T4. Random, GP, PFN lệch nhiều (GP: 0.79 so với 2.49).** [không rõ] Bài báo không định nghĩa "loss phân loại" cho các phương pháp không có bộ phân loại, và code tác giả không có phần đánh giá baseline cho Table 1. Dự án tự định nghĩa: mỗi phương pháp cho ra `P(f(x) < τ)` trên mẫu held-out. Hai bên không so được. Riêng LFBO, vốn là bộ phân loại, có Avg khớp (0.582 so với 0.589).

**T5. Sai số chuẩn của bài báo nhỏ hơn nhiều.** [đã đo, phía dự án] Độ lệch chuẩn giữa các seed của original là 0.0053, nên sai số chuẩn trên 100 lần chạy sẽ khoảng 0.0005; bài báo ghi 0.00003, nhỏ hơn 18 lần. Với MALIBO chênh khoảng 7 lần. Tỉ lệ ±/trung bình của bài báo gần như bằng nhau ở mọi phương pháp (0.11–0.17%). Có thể họ giữ cố định tập task và chỉ đổi một nguồn ngẫu nhiên nhỏ; tôi không xác định được.

**T6. Văn bản và bảng không thống nhất.** Bài báo ghi "normalized regret là thước đo chính" nhưng Table 1 là loss phân loại. Bài báo ghi giảm 97.9% so với GP; tính từ bảng là 97.7% (97.9% là so với Random).

---

## 4. Ablation

Bài báo (so với MALIBO): task-only −21.7% / −28.1%; sample-only −71.8% / −70.5%; đầy đủ −95.7% / −88.6%.

**A1. Code tác giả.** [đã đo, 2 seed] Xem bảng ở mục 3.2.

- Task-only (`--use_dro`): 0.333 / 0.595, khớp giá trị suy ra từ bài báo trong 1–2%.
- Sample-only (`--use_kdpp`): 0.018 / 0.094, **không khớp** 0.121 / 0.242 của bài báo, mà gần như trùng hàng MetaGameBO đầy đủ. Tức là với code đã công bố, toàn bộ mức giảm đến từ `_apply_kdpp_weights`; bật thêm DRO không đổi gì đáng kể. Tôi không biết cấu hình nào cho ra −71.8%.
- Khi tắt hẳn k-DPP, DRO không cải thiện Avg (0.558 so với 0.549) và giảm CVaR khoảng 12%, nhưng hai seed lệch nhau nhiều (0.658 và 0.458) nên chưa kết luận được.

**A2. Remake.** [đã đo, 3 seed] `results/forrester_remake_ablation.md`:

| Biến thể | Nhãn LFBO: Avg | Nhãn LFBO: CVaR | Nhãn theo bài báo: Avg | Nhãn theo bài báo: CVaR |
|---|---|---|---|---|
| MALIBO | 0.528 | 1.001 | 0.128 | 0.617 |
| + CVaR (task-only) | 0.474 (−10%) | 0.844 (−16%) | 0.106 (−18%) | 0.496 (−20%) |
| + k-DPP và Eq. 6 (sample-only) | 0.677 (+28%) | 1.003 (0%) | 0.173 (+35%) | 0.812 (+32%) |
| Đầy đủ | 0.611 (+16%) | 0.949 (−5%) | 0.115 (−11%) | 0.500 (−19%) |

- **Mức task cùng chiều với bài báo** nhưng nhỏ hơn (−10 đến −20% so với −22 đến −28%).
- **Phần lớn lợi ích của mức task là do ít overfit hơn.** Mỗi bước chỉ huấn luyện 13–17% số task, nên mô hình overfit chậm hơn MALIBO. Nếu lấy vòng lặp có val loss thấp nhất (tương đương early stopping), các biến thể gần bằng nhau: 0.431 / 0.704 (MALIBO) so với 0.441 / 0.715 (task-only).
- **Mức sample làm tệ đi ở cả hai cách gán nhãn**, ngược với bài báo. Với nhãn LFBO, train loss dừng ở 0.699 ≈ ln 2, tức bộ phân loại đoán 0.5 ở mọi điểm. Lý do: Eq. 6 tăng trọng số cho mẫu có loss cao, mà với nhãn LFBO đó chính là các cặp mâu thuẫn (cùng một điểm mang cả nhãn 0 và 1) không thể khớp được.
- Kết quả mới có 3 seed.

---

## 5. Figure 3 (HPOBench, NAS-Bench-201, HPO-B)

### 5.1 Regret ở bước cuối

Số của bài báo và của MALIBO đọc từ hình [quan sát hình].

**NAS-Bench-201** (dự án: 15 lần chạy, đọc từ hình Kaggle):

| Phương pháp | MetaGameBO Fig. 3 | MALIBO Fig. 5 | Dự án |
|---|---|---|---|
| ABLR | 2.8e-2 | 2.9e-2 | 3.3e-3 |
| GP | 1.9e-2 | 1.9e-2 | 3.3e-3 |
| Random | 1.1e-2 | 1.15e-2 | 6.5e-3 |
| RGPE | 8e-3 | 8e-3 | 5.5e-5 |
| PFN | 7.5e-3 | 7.5e-3 | 0 (từ bước ~165) |
| LFBO | 1.1e-3 | 1.1e-3 | 2.7e-4 |
| MALIBO | 9.5e-4 | 9e-4 | 0 (từ bước ~270) |
| MetaGameBO | 8.5e-4 | – | original: 0 (~230); remake: 0 (~245) |

**HPOBench** (dự án: 20 lần chạy; GP, LFBO, RGPE, remake chưa có trên máy local):

| Phương pháp | MetaGameBO Fig. 3 | MALIBO Fig. 5 | Dự án |
|---|---|---|---|
| ABLR | 1.4e-5 | 3e-5 | 4.9e-6 |
| Random | 4e-6 | 8e-6 | 9.7e-6 |
| PFN | 2.5e-6 | 5.5e-6 | 0 |
| MALIBO | 7e-9 | 3e-9 | 2.6e-8 |
| MetaGameBO | 5e-9 | – | original: 8.7e-9 |

**HPO-B** (dự án: bản continuous, 430 lần chạy):

| Phương pháp | MetaGameBO Fig. 3 (trục ghi 500) | MALIBO Fig. 5 (bước 100) | Dự án, bước 105 | Dự án, bước 500 |
|---|---|---|---|---|
| Random | 4.8e-2 | 4.9e-2 | – | – |
| LFBO | 3.5e-2 | 3.6e-2 | – | – |
| RGPE | 2.7e-2 | 2.8e-2 | – | – |
| ABLR | 2.2e-2 | 2.3e-2 | 3.5e-2 | 2.4e-2 |
| PFN | 2.1e-2 | 2.1e-2 | 3.9e-2 | 3.0e-2 |
| GP | 1.7e-2 | 1.7e-2 | – | – |
| MALIBO | 1.1e-2 | 8.7e-3 | 2.4e-2 | 1.2e-2 |
| MetaGameBO | 8.3e-3 | – | (mới 3/16 space) | |

### 5.2 Các lệch và lý do

**F1. Baseline trong Figure 3 có vẻ lấy từ bài báo MALIBO, không chạy lại.** [quan sát hình]

- Ở NAS-Bench-201, giá trị cuối của cả 7 baseline trùng với Figure 5 của MALIBO (bảng trên).
- Ở HPO-B, giá trị cuối của Random, LFBO, RGPE, ABLR, PFN, GP ở "bước 500" trùng với giá trị của MALIBO ở **bước 100**. Hình của MALIBO chỉ có 100 bước. Các điểm đặc trưng cũng trùng sau khi nhân trục x với 5: RGPE tụt mạnh ở bước ~55 trong hình MALIBO và ở ~275 trong hình này; GP cắt xuống dưới ABLR ở ~65 và ~325.
- Ở HPOBench, các baseline thấp hơn hình MALIBO một hệ số gần như không đổi, khoảng 2.1 lần. Tôi không biết lý do.
- Đây là so sánh bằng mắt. Bạn có thể tự kiểm tra bằng cách đặt hai hình cạnh nhau: Figure 3 của `paper.pdf` và Figure 5 của arXiv 2307.03565v3.

Hệ quả: baseline của bài báo được tạo bằng cài đặt của nhóm MALIBO (theo bài MALIBO: GP của BoTorch, code gốc của LFBO, PFN, ...), trên pipeline HPOBench và NAS-Bench-201 không công khai, với 100 lần chạy. Baseline tự cài của dự án không thể khớp từng đường.

**F2. Regret của dự án chạm đúng 0, của bài báo dừng ở khoảng 1e-3.** [đã đo, phía dự án]

- Bảng tra của dự án không nhiễu (trung bình các seed huấn luyện), nên tìm đúng cấu hình tốt nhất cho regret bằng 0.
- Với 15 lần chạy, chỉ cần cả 15 lần đều tìm ra là trung bình bằng 0. Bài báo lấy trung bình trên số lần chạy lớn hơn nhiều, nên luôn còn một phần chưa tìm ra.
- Thang đo ở NAS-Bench-201: trên CIFAR-100 chỉ có **một** kiến trúc có regret dưới 1e-3 (kiến trúc thứ nhì đã là 4.2e-3); trên ImageNet16 có hai. Mức 1e-3 của bài báo vì vậy nghĩa là phần lớn các lần chạy tìm ra tối ưu và một phần nhỏ dừng ở hạng 2–10.

**F3. PFN: bài báo 7.5e-3, dự án về 0 nhanh nhất.** [không rõ] Dự án dùng mô hình HEBO+ của PFNs4BO, chấm EI trực tiếp trên các điểm của bảng, mã hoá 6 chiều thứ tự. Cách nhóm MALIBO chạy PFN (mã hoá, cách tối ưu acquisition) không được nêu. Đây là lệch lớn nhất về thứ hạng.

**F4. Random tốt hơn GP và ABLR trong bài báo; trong dự án thì ngược lại.** Cùng nguyên nhân với F1 và F3: cài đặt baseline khác. GP của dự án là Matern-5/2 ARD của scikit-learn trên mã hoá thứ tự.

**F5. HPO-B không so được.** [đọc code] Dự án dùng surrogate liên tục và 500 bước; bài báo dùng bản discrete, và theo F1 nhiều khả năng là 100 bước. Thêm nữa:

- 11 trong 86 dataset test có dưới 505 đánh giá (ví dụ space 4796: 300; 5860: 200–389). Ở bản discrete, các dataset này bị vét hết trước bước 500.
- Ở bản continuous, 52–73% số lần chạy của dự án kết thúc ở regret đúng bằng 0, vì dự đoán của surrogate vượt `y_max` trong thống kê và bị kẹp.
- Code tác giả huấn luyện trên toàn bộ meta-train; dự án lấy 512 điểm mỗi dataset.

**F6. Original, remake và MALIBO gần như bằng nhau trong dự án.** Ở NAS-Bench-201 cả ba về 0 trong khoảng bước 230–270; ở HPOBench original 8.7e-9 và MALIBO 2.6e-8, với 85–95% số lần chạy đã về đúng 0. Với 15–20 lần chạy, dự án chưa đủ để khẳng định hay bác bỏ câu "consistently outperforms" của bài báo.

**F7. Mức task gần như vô nghĩa trên HPOBench và NAS-Bench-201.** [đọc code] Leave-one-task-out chỉ để lại 3 và 2 task nguồn. "20% task khó nhất" của 2 task là đúng một task, ở cả remake lẫn code tác giả (`max(1, int(2·0.2)) = 1`). Cơ chế CVaR chỉ có ý nghĩa ở Forrester (200 task) và HPO-B (20–60 task).

**F8. Ba ô đường huấn luyện trong hình NAS-Bench-201 của dự án.** [đã đo một phần]

- *MALIBO tăng từ 0.56 lên 2.1.* Không có early stopping, và cách chia held-out theo mẫu tách hai bản sao mang nhãn trái nhau của cùng một điểm tốt sang hai phía. Mô hình càng khớp train thì càng sai trên bản sao còn lại.
- *Remake đi ngang ở ~0.67 ≈ ln 2.* Cùng hiện tượng với sample-only ở mục 4: bộ phân loại gần như đoán 0.5.
- *Original thấp nhất (0.45–0.57).* Trên NAS-Bench-201, `_apply_kdpp_weights` giảm tỉ trọng lớp tốt xuống 12% (HPOBench 9%, HPO-B 14–31%), nên loss của nó đo trên thang dễ hơn. Ba đường này không so được với nhau.

**F9. Nguồn dữ liệu.** Dự án dùng bản chuyển đổi của Syne Tune (link gốc đã chết), mục tiêu là sai số validation ở epoch cuối, trung bình các seed. MALIBO ghi "highest validation accuracy" cho NAS-Bench-201; tôi không biết họ lấy epoch nào.

---

## 6. Những điểm không kiểm chứng được

- Mọi siêu tham số và chi tiết benchmark nằm ở appendix, mà appendix không có.
- Cách tính "loss" cho Random, GP, LFBO, PFN ở Table 1.
- Nguồn gốc của sai số chuẩn trong Table 1 và của con số "100 independent runs". Tác giả cho biết hồ sơ thí nghiệm gốc đã mất.
- Cấu hình cho ra mức −71.8% / −70.5% của sample-only.
- Script cho HPOBench và NAS-Bench-201: cả code tác giả lẫn code MALIBO đều không có.
- Hai script công bố không chạy được nguyên trạng: `run_forrester.py` gọi `MALIBO(...)` nhưng chỉ import `MetaGameBO`; `run_hpob.py` import module `malibo` không tồn tại.
- Bài báo nhắc FSBO trong phần nhận xét nhưng Figure 3 không có FSBO. FSBO có trong hình của MALIBO.

---

## 7. Khuyến nghị

1. **Table 1 và ablation:** trình bày ba cột cạnh nhau: số của bài báo, số do code tác giả báo, và số đo lại bằng trọng số Eq. 3 gốc. Nêu rõ thước đo phụ thuộc trọng số đánh giá. Bảng ở mục 3.2 dùng được ngay.
2. **HPO-B:** chạy lại bằng `--discrete --budget 105` thay cho bản continuous. Bản này cũng nhanh hơn vì không gọi surrogate.
3. **Figure 3:** không kỳ vọng khớp từng baseline; so thứ hạng và bậc độ lớn. Nêu F1 như một quan sát kèm cách kiểm tra.
4. **Số seed:** tăng lên ít nhất 10–20 cho NAS-Bench-201 và HPOBench, hoặc dùng `--noisy`, để regret không chạm 0.
5. **Tuỳ chọn:** thêm early stopping cho MALIBO để gần code MALIBO gốc. Việc này buộc chạy lại MALIBO ở mọi benchmark.

---

## 8. Tái lập các số chẩn đoán

```bash
python -m experiments.author_forrester_ablation               # mục 3.2, A1: 12 lần chạy, 4–7 phút mỗi lần
python -m experiments.remake_forrester_ablation               # mục 4, A2: 24 lần chạy, 1–3 phút mỗi lần
python -m experiments.analyze_hpob --discrete --budget 105    # HPO-B theo giao thức của code tác giả
```

Kết quả đã có sẵn trong `results/forrester_author_ablation.md` và `results/forrester_remake_ablation.md`; chạy lại lệnh sẽ dùng cache trong `results/raw/`.

Những gì **chưa** được kiểm chứng:

- Ablation code tác giả mới có 2 seed, chạy trên CPU; bài báo có thể dùng GPU.
- Ablation remake mới có 3 seed.
- `--discrete` mới thử trên space 5860 với Random và LFBO.
- Mọi so sánh với hình của bài báo và của MALIBO là đọc bằng mắt.

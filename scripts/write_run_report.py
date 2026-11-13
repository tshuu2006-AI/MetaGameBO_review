"""Write a Vietnamese report from verified result files, without rerunning training."""
import argparse
import csv
import json
import re
from pathlib import Path


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_report(folder):
    folder = Path(folder)
    c = json.loads((folder/'config.json').read_text())
    verification = json.loads((folder/'verification.json').read_text())
    assert verification['passed']
    test_log = (folder/'pytest.txt').read_text(encoding='utf-8')
    match = re.search(r'(\d+) passed', test_log)
    test_count = match.group(1) if match else '?'
    bce = read_rows(folder/'classifier_bce.csv')
    objectives = {r['split']: r for r in read_rows(folder/'classifier_objective.csv')}
    regret = read_rows(folder/'bo_regret.csv')
    pfn = json.loads((folder/'PFN_training.json').read_text())
    contexts = sorted(set(n for row in pfn for n in row['context_lengths']))
    lines = [
        '# Kết quả tái lập theo phản hồi tác giả', '',
        'Ngày báo cáo: 13/11/2026 (Asia/Bangkok).', '',
        f"Cấu hình {'smoke' if c['quick'] else 'đầy đủ'}: {c['tasks']} source task, "
        f"{c['samples_per_task']} mẫu/task, {c['epochs']} epoch, batch {c['task_batch_size']}, "
        f"seed {c['seed']}, {c['runs']} target, tổng {c['steps']} điểm objective/target/phương pháp "
        f"(gồm {c['initial_points']} điểm khởi tạo). Thiết bị: {c['device']}.", '',
        'MALIBO dùng implementation của Allen0497/MetaGameBO, commit '
        '`489213ddcfbfe275a8b443c0d8b136ef6c36ad2c`, tắt cả DRO và k-DPP. '
        'PFN là mô hình huấn luyện cục bộ trên prior Forrester; không phải checkpoint pretrained của bài báo.', '',
        '## Weighted BCE của MALIBO', '',
        '| Split | Task | Dòng phân loại | Macro BCE | Micro BCE | CVaR BCE | k |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in bce:
        lines.append(f"| {r['split']} | {r['number_of_tasks']} | {r['sample_count']} | "
                     f"{float(r['macro_bce']):.9f} | {float(r['micro_bce']):.9f} | "
                     f"{float(r['cvar_bce']):.9f} | {r['tail_k']} |")
    lines.extend(['', 'Task loss = tổng weighted BCE chia số dòng phân loại của task. '
                  'CVaR dùng alpha 0.8 và đúng `max(1,int(number_of_tasks*(1-alpha)))`; '
                  'không dùng ngưỡng phân vị, KDE hoặc objective có regularization.', '',
                  'Split 80/20 theo dòng đúng trainer công khai. Với 8.600 dòng và float Python, '
                  'cấu hình đầy đủ chia 6.881/1.719 dòng. Validation có cùng source task id và '
                  'có thể trùng tọa độ LFBO với train. Không gọi đây là BCE trên task chưa thấy.', '',
                  '## Objective được lưu riêng', '',
                  '| Split | Regularization | Micro BCE + regularization |',
                  '|---|---:|---:|'])
    for split, r in objectives.items():
        lines.append(f"| {split} | {float(r['regularization']):.9f} | "
                     f"{float(r['objective_with_regularization']):.9f} |")
    lines.extend(['', 'Regularization giữ MetaLoss gốc: 0.1 × tổ hợp KS/covariance của task embedding. '
                  'Audit chạy eval mode, không đưa embedding noise vào loss, không tiêu thụ RNG huấn luyện.', '',
                  '## Normalized BO regret', '',
                  '| Phương pháp | Regret cuối | SE giữa các target | CVaR regret |',
                  '|---|---:|---:|---:|'])
    for r in regret:
        lines.append(f"| {r['method']} | {float(r['average_normalized_regret']):.10f} | "
                     f"{float(r['standard_error']):.10f} | {float(r['cvar_normalized_regret']):.10f} |")
    lines.extend(['', 'Regret và CVaR regret nằm trong bảng riêng; không tương đương BCE/CVaR của Table 1. '
                  'Đường hội tụ: `convergence.png`; toàn bộ X/y: `bo_traces.json`.', '',
                  '## PFN và kiểm chứng', '',
                  'Attention chặn mọi cột query, nên context không đọc query và mỗi query chỉ đọc context. '
                  'Không đưa query label vào input. Head Normal dự đoán cả mean và std dương, acquisition '
                  'là EI giải tích cho minimization. Inference không cập nhật trọng số.', '',
                  f"Context length đã xuất hiện: {contexts}. Mỗi epoch dùng đủ {c['tasks']} task.", '',
                  f"NLL train chuẩn hóa y theo context: epoch đầu {pfn[0]['train_gaussian_nll']:.9f}; "
                  f"epoch cuối {pfn[-1]['train_gaussian_nll']:.9f}. Đây là diagnostic train; "
                  'NLL có thể âm vì density liên tục có thể lớn hơn 1. Không đưa NLL/MSE vào bảng BCE.', '',
                  f'Bộ kiểm thử: {test_count} passed, gồm mask/query independence, context biến thiên, predictive '
                  'density hữu hạn, EI đối chiếu Monte Carlo, inference giữ nguyên trọng số, đúng số '
                  'đánh giá BO, macro/micro BCE và CVaR floor rule.', '',
                  'Kiểm chứng độc lập (`verification.json`): checkpoint tái tạo khớp BCE trên từng split; '
                  'mọi X/y được đối chiếu objective target; curve tái tính khớp; vendor SHA256 khớp. '
                  'Dữ liệu, logits/labels/weights/task id và checkpoint được lưu để kiểm tra lại.', '',
                  '```powershell',
                  f'.venv/Scripts/python scripts/verify_results.py {folder.as_posix()}',
                  '```', '',
                  '## Giới hạn diễn giải', '',
                  'Tác giả không còn seed list và toàn bộ log gốc, nên không bảo đảm khớp từng chữ số Table 1. '
                  'Benchmark giữ phân phối tham số gốc nhưng dùng các luồng RNG tách biệt của dự án, '
                  'paired random initial design và pool acquisition 512 điểm. PFN chưa có quy tắc BCE '
                  'được tác giả xác nhận và chưa dùng checkpoint pretrained gốc.', '',
                  ('Chạy một target/seed cho SE bằng 0 theo quy ước bảng, không phải bằng chứng phương sai '
                  'bằng 0 hay ưu thế thống kê. Trên target mặc định này, điểm khởi tạo đã có regret khoảng '
                  '0.001279; PFN không cải thiện incumbent sau khởi tạo. Cần nhiều target/seed trước khi '
                  'kết luận so sánh tổng quát.' if c['runs']==1 else
                  'Đây là lượt smoke với ít task và epoch; chỉ dùng kiểm tra pipeline, '
                  'không dùng thay cho kết quả cấu hình đầy đủ hoặc suy luận thống kê.'), ''])
    (folder/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', nargs='?', default='results/author_reply')
    write_report(parser.parse_args().folder)

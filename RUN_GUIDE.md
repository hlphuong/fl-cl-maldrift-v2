# Hướng Dẫn Chạy FL-CL-MalDrift v2

Tài liệu này hướng dẫn cách cài đặt, chuẩn bị dữ liệu, chạy so sánh 3 phương pháp, chạy K-Fold, xuất bảng kết quả, và giải thích vai trò các file chính trong project.

## 1. Cấu Trúc Thư Mục

```text
FL_CL_MalDrift_v2/
|-- CSV/                                  # CSV gốc từ CICMalDroid
|-- FL_CL_MalDrift_Summary_1.docx         # file tóm tắt/yêu cầu thiết kế
|-- RUN_GUIDE.md                          # hướng dẫn chạy project
|-- EXPERIMENT_EXPLANATION.md             # giải thích luồng thí nghiệm
|-- fl_cl_maldrift_v2/
|   |-- main.py                           # file chính để chạy training/testing
|   |-- config.py                         # cấu hình hyperparameter tập trung
|   |-- prepare_cicmaldroid.py            # tiền xử lý CICMalDroid
|   |-- requirements.txt                  # danh sách thư viện Python
|   |-- data/
|   |   |-- dataset.py                    # load data, chia task, chia client
|   |   |-- cicmaldroid/features.csv      # dataset đã tiền xử lý
|   |-- fl/
|   |   |-- client.py                     # logic client FL + train local + DRC
|   |   |-- server.py                     # FedAvg/drift-aware aggregation
|   |-- models/mlp.py                     # mô hình MalwareMLP
|   |-- modules/
|   |   |-- drift_detector.py             # drift detector
|   |   |-- drc.py                        # Drift Resolution Controller
|   |   |-- continual_learning.py         # Replay Buffer + EWC
|   |   |-- privacy.py                    # DP-SGD
|   |-- utils/
|   |   |-- metrics.py                    # ACC/F1/Recall/Forget/BWT/FWT
|   |   |-- logger.py                     # lưu JSON/CSV/PNG
|   |-- scripts/
|   |   |-- export_kfold_task_table.py    # xuất bảng tổng hợp K-Fold
|   |-- results/                          # kết quả thực nghiệm
```

## 2. Cài Đặt Môi Trường

Di chuyển vào thư mục code:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2
```

Nếu đã có môi trường `malenv`, kiểm tra:

```powershell
.\malenv\Scripts\python.exe --version
```

Nếu clone lại từ GitHub và chưa có môi trường:

```powershell
python -m venv malenv
.\malenv\Scripts\python.exe -m pip install --upgrade pip
.\malenv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Chuẩn Bị Dataset CICMalDroid

Script `prepare_cicmaldroid.py` đọc CSV gốc từ thư mục `..\CSV`, chọn top 200 feature bằng mutual information, encode label, rồi lưu ra:

```text
fl_cl_maldrift_v2/data/cicmaldroid/features.csv
```

Tạo dataset malware:benign = 1:1:

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 1.0
```

Tạo dataset malware:benign = 2:1:

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 2.0
```

Dataset 2:1 hiện tại có:

```text
Tổng mẫu: 5385
Benign:  1795
Malware: 3590

Class 1 Adware:   479
Class 2 Banking:  781
Class 3 SMS:      1405
Class 4 Riskware: 925
Class 5 Benign:   1795
```

## 4. Chạy So Sánh 3 Phương Pháp

Ba phương pháp được so sánh:

```text
FedAvg:
  Federated Learning cơ bản, không xử lý drift, không continual learning.

FL-MalDrift:
  Có drift-aware aggregation, nhưng không có Replay/EWC.

FL-CL-MalDrift:
  Có drift-aware aggregation, DRC, Replay Buffer và EWC.
```

Lệnh so sánh chính với 5 task, 5 client:

```powershell
.\malenv\Scripts\python.exe main.py --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_strict --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --out results\compare_5clients_strict_2to1
```

Ý nghĩa tham số:

```text
--compare
  Chạy cả 3 phương pháp.

--dataset cicmaldroid
  Dùng dataset CICMalDroid đã tiền xử lý.

--tasks 5
  Chia dữ liệu thành 5 task liên tục.

--rounds 25
  Mỗi task chạy 25 communication rounds.

--clients 5
  Dùng 5 client trong federated learning.

--task_strategy category_strict
  Chia task theo từng họ mã độc rõ ràng để tạo concept drift mạnh.

--partition_strategy category
  Chia train data cho client theo kiểu non-IID/category specialization.

--local_epochs 3
  Mỗi client train local 3 epoch trước khi gửi update lên server.

--tau_init 0.25
  Ngưỡng drift ban đầu của server.

--tau_min 0.2
  Ngưỡng drift thấp nhất.

--warmup_rounds 0
  Không dùng warmup, server xét drift ngay từ round đầu.

--ewc_lambda 0.1
  Độ mạnh của EWC.

--replay_buffer 800
  Kích thước replay buffer.
```

## 5. Chạy K-Fold 5 Folds, 5 Clients, 3 Phương Pháp

Đây là lệnh nên dùng để lấy kết quả ổn định nhất:

```powershell
.\malenv\Scripts\python.exe main.py --kfold 5 --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_strict --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --out results\kfold5_clients5_compare
```

Luồng chạy:

```text
Fold 1 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4
Fold 2 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4
Fold 3 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4
Fold 4 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4
Fold 5 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4
```

Trong mỗi fold, chương trình chạy đủ 3 phương pháp:

```text
FedAvg
FL-MalDrift
FL-CL-MalDrift
```

Các file kết quả chính:

```text
results/kfold5_clients5_compare/kfold_per_fold.csv
results/kfold5_clients5_compare/kfold_per_task.csv
results/kfold5_clients5_compare/kfold_summary.csv
```

## 6. Xuất Bảng Tổng Hợp CSV Và PNG

Sau khi chạy K-Fold, xuất bảng tổng hợp theo format `Scenario / Task / Method`:

```powershell
.\malenv\Scripts\python.exe scripts\export_kfold_task_table.py --result_dir results\kfold5_clients5_compare --out_prefix kfold_task_summary_table_full
```

File xuất ra:

```text
results/kfold5_clients5_compare/kfold_task_summary_table_full.csv
results/kfold5_clients5_compare/kfold_task_summary_table_full.png
```

Bảng gồm:

```text
Scenario = Fold 1..Fold 5 và Mean ± Std
Task     = Task 0..Task 4, Overall
Method   = FedAvg, FL-MalDrift, FL-CL-MalDrift
Metric   = Acc, Prec, Rec, F1, Forget, BWT, FWT
```

Lưu ý: `Forget`, `BWT`, `FWT` là metric của cả chuỗi continual learning, nên chỉ có ở dòng `Overall`.

## 7. Chạy DRC Stress Test

Lệnh này dùng để kiểm tra cơ chế drift loop, escalation và recovery:

```powershell
.\malenv\Scripts\python.exe main.py --method fl_cl_maldrift --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_strict --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --drc_stress --out results\drc_stress_5clients_2to1
```

Cách đọc:

```text
EscRate > 0
  Có escalation xảy ra.

RecRate > 0
  Client có thể recovery và quay lại federation.
```

## 8. Vai Trò Các File Chính

| File | Vai trò |
|---|---|
| `main.py` | Parse CLI, tạo config, chạy task, chạy compare, chạy K-Fold, in summary. |
| `config.py` | Chứa hyperparameter: FL, DATA, DRIFT, DRC, CL, SERVER. |
| `prepare_cicmaldroid.py` | Tiền xử lý CICMalDroid, hỗ trợ `--malware_ratio`. |
| `data/dataset.py` | Load dataset, chia task, chia client non-IID. |
| `models/mlp.py` | Định nghĩa mô hình MalwareMLP. |
| `fl/client.py` | Client FL: train local, tính drift score, gọi DRC, trả update. |
| `fl/server.py` | Server FL: FedAvg, drift-aware filtering, drift-aware weighting, cập nhật `tau`. |
| `modules/drift_detector.py` | Tạo drift score trong khoảng `[0,1]`. |
| `modules/drc.py` | Drift Resolution Controller: STABLE, REPLAY, EWC, ESCALATION, RECOVERY. |
| `modules/continual_learning.py` | Replay Buffer và EWC. |
| `utils/metrics.py` | Tính ACC, Precision, Recall, F1, Forgetting, BWT, FWT. |
| `utils/logger.py` | Lưu JSON, CSV, PNG kết quả. |
| `scripts/export_kfold_task_table.py` | Xuất bảng tổng hợp K-Fold thành CSV và PNG. |


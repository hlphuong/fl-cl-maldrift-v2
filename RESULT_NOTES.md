# Nhận Xét Chia Dữ Liệu, Lệnh Chạy Và Kết Quả

Tài liệu này tổng hợp lại các nhận xét quan trọng sau khi đã thử nhiều cách chia dữ liệu trong thực nghiệm FL-CL-MalDrift trên CICMalDroid.

## 1. Các Cách Chia Dữ Liệu Và Ảnh Hưởng Đến Kết Quả

### 1.1. Dataset malware:benign = 1:1

Lệnh tạo dataset:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2

.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 1.0
```

Số lượng dữ liệu:

```text
Malware = 1795
Benign  = 1795
Total   = 3590
```

Nhận xét:

Với tỷ lệ 1:1, dữ liệu malware và benign cân bằng. Mô hình học hai lớp tương đối đều, nên Precision thường khá ổn vì mô hình không quá thiên về dự đoán malware. Tuy nhiên, trong bài toán phát hiện malware, tỷ lệ này có thể làm Recall chưa cao bằng trường hợp tăng số lượng malware, vì mô hình nhìn thấy ít mẫu malware hơn.

Kết quả 1:1 trước đó cho thấy FL-CL-MalDrift cải thiện ACC/F1/Recall, nhưng phần Forget chưa thể hiện rõ vì dữ liệu chưa tạo đủ áp lực drift giữa các task.

### 1.2. Dataset malware:benign = 2:1

Lệnh tạo dataset:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2

.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 2.0
```

Số lượng dữ liệu hiện tại:

```text
Malware = 3590
Benign  = 1795
Total   = 5385
```

Phân phối class:

| Class | Loại | Số mẫu |
|---:|---|---:|
| 1 | Adware | 479 |
| 2 | Banking | 781 |
| 3 | SMS | 1405 |
| 4 | Riskware | 925 |
| 5 | Benign | 1795 |

Nhận xét:

Khi tăng malware lên tỷ lệ 2:1, mô hình được học nhiều mẫu malware hơn nên Recall và F1 thường tăng. Điều này phù hợp với bài toán malware detection vì bỏ sót malware nguy hiểm hơn báo nhầm benign.

Tuy nhiên, Precision có thể giảm nhẹ trong một số lần chạy. Lý do là mô hình có xu hướng dự đoán malware mạnh hơn, dẫn đến một phần benign bị báo nhầm là malware. Vì vậy khi malware tăng, thường thấy:

```text
Recall tăng
F1 tăng
ACC có thể tăng
Precision có thể dao động hoặc giảm nhẹ
```

## 2. Các Cách Chia Task Đã Thử

### 2.1. `category_strict`

Kiểu chia cũ:

```text
Task 0: chủ yếu Adware
Task 1: chủ yếu Banking
Task 2: chủ yếu SMS
Task 3: chủ yếu Riskware
Task 4: mixed/held-out
```

Ưu điểm:

Kiểu chia này tạo drift rất mạnh, dễ thấy catastrophic forgetting.

Hạn chế:

Một số task gần như chỉ đại diện cho một họ malware, nên chưa thật tự nhiên. Trong thực tế, ở một thời điểm hệ thống thường vẫn có nhiều họ malware cùng tồn tại, chỉ khác tỷ trọng từng họ.

### 2.2. `category_dominant` với `dominant_ratio=0.75`

Lệnh K-Fold:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2

.\malenv\Scripts\python.exe main.py --kfold 5 --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_dominant --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --out results\kfold5_clients5_dominant
```

Ý nghĩa:

Mỗi task đều có đủ 4 họ malware gồm Adware, Banking, SMS và Riskware. Trong mỗi task, một họ chiếm đa số ở mức vừa phải. Đây là cách chia hợp lý hơn `category_strict` vì vẫn mô phỏng được việc nhiều họ malware cùng tồn tại.

Ảnh hưởng đến kết quả:

Do task nào cũng còn có dữ liệu của các họ malware cũ, catastrophic forgetting bị giảm nhẹ. Vì vậy FL-CL-MalDrift vẫn tốt hơn, nhưng khoảng cách Forget/BWT giữa các phương pháp chưa quá lớn.

### 2.3. `category_dominant` với `dominant_ratio=0.90`

Lệnh K-Fold mới nhất:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2

.\malenv\Scripts\python.exe main.py --kfold 5 --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_dominant --dominant_ratio 0.90 --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --out results\kfold5_clients5_dominant_r090
```

Ý nghĩa:

Mỗi task vẫn có đủ 4 họ malware, nhưng họ chiếm đa số sẽ áp đảo hơn. Ví dụ task Adware-dominant vẫn có Banking/SMS/Riskware, nhưng số lượng các họ này ít hơn nhiều.

Phân phối dữ liệu khi test split:

| Task | Họ chiếm đa số | Adware | Banking | SMS | Riskware | Benign |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Adware | 431 | 20 | 47 | 23 | 260 |
| 1 | Banking | 12 | 703 | 47 | 23 | 393 |
| 2 | SMS | 12 | 20 | 632 | 23 | 343 |
| 3 | Riskware | 12 | 19 | 47 | 833 | 456 |
| 4 | SMS | 12 | 19 | 632 | 23 | 343 |

Ảnh hưởng đến kết quả:

Đây là cấu hình làm rõ catastrophic forgetting tốt nhất hiện tại. Vì task mới có phân phối khác mạnh hơn task cũ, FedAvg và FL-MalDrift dễ bị giảm hiệu năng trên task cũ. FL-CL-MalDrift có Replay Buffer và EWC nên giữ lại tri thức cũ tốt hơn.

## 3. Lệnh Nào Chạy Cái Gì

| Mục đích | Lệnh | Kết quả chính |
|---|---|---|
| Tạo dataset cân bằng 1:1 | `prepare_cicmaldroid.py --force --malware_ratio 1.0` | Ghi lại `data/cicmaldroid/features.csv` với 1795 malware và 1795 benign |
| Tạo dataset malware nhiều hơn 2:1 | `prepare_cicmaldroid.py --force --malware_ratio 2.0` | Ghi lại `data/cicmaldroid/features.csv` với 3590 malware và 1795 benign |
| So sánh nhanh 3 phương pháp | `main.py --compare ...` | Chạy FedAvg, FL-MalDrift, FL-CL-MalDrift một lần và in `COMPARISON SUMMARY` |
| So sánh ổn định bằng 5-fold | `main.py --kfold 5 --compare ...` | Sinh `kfold_per_fold.csv`, `kfold_per_task.csv`, `kfold_summary.csv` |
| Tạo drift/forgetting mạnh hơn | Thêm `--dominant_ratio 0.90` | Forget và BWT thể hiện rõ hơn |
| Xuất bảng CSV/PNG | `scripts/export_kfold_task_table.py --result_dir ...` | Sinh bảng tổng hợp `.csv` và `.png` |
| Kiểm tra drift loop | Thêm `--drc_stress` | Dùng để quan sát escalation/recovery, không nên đánh giá chỉ bằng ACC/F1 |

Lệnh xuất lại bảng CSV/PNG khi đã có kết quả mới:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2

.\malenv\Scripts\python.exe scripts\export_kfold_task_table.py --result_dir results\kfold5_clients5_dominant_r090 --out_prefix kfold_task_summary_table_full
```

File sinh ra:

```text
results\kfold5_clients5_dominant_r090\kfold_task_summary_table_full.csv
results\kfold5_clients5_dominant_r090\kfold_task_summary_table_full.png
```

## 4. Đánh Giá Kết Quả Gần Đây Nhất

Kết quả mới nhất dùng:

```text
Dataset: CICMalDroid 2:1
K-Fold: 5 folds
Clients: 5
Tasks: 5
Task strategy: category_dominant
Dominant ratio: 0.90
Rounds: 25
Local epochs: 3
Replay buffer: 800
EWC lambda: 0.1
```

Bảng kết quả:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.8802 ± 0.0127 | 0.9159 ± 0.0042 | 0.9070 ± 0.0116 | 0.9039 ± 0.0231 | 0.0407 ± 0.0202 | -0.0382 ± 0.0211 | 0.3780 ± 0.0124 |
| FL-MalDrift | 0.8829 ± 0.0131 | 0.9138 ± 0.0053 | 0.9098 ± 0.0116 | 0.9110 ± 0.0225 | 0.0361 ± 0.0155 | -0.0321 ± 0.0178 | 0.3768 ± 0.0137 |
| FL-CL-MalDrift | 0.9089 ± 0.0101 | 0.9164 ± 0.0031 | 0.9320 ± 0.0087 | 0.9511 ± 0.0183 | 0.0108 ± 0.0126 | -0.0050 ± 0.0157 | 0.3812 ± 0.0183 |

### 4.1. Đánh giá theo khả năng phát hiện malware

FL-CL-MalDrift đạt kết quả tốt nhất ở ACC, F1 và Recall:

```text
ACC:
  FedAvg          = 0.8802
  FL-MalDrift     = 0.8829
  FL-CL-MalDrift  = 0.9089

Recall:
  FedAvg          = 0.9039
  FL-MalDrift     = 0.9110
  FL-CL-MalDrift  = 0.9511

F1:
  FedAvg          = 0.9070
  FL-MalDrift     = 0.9098
  FL-CL-MalDrift  = 0.9320
```

Điều này cho thấy FL-CL-MalDrift không chỉ giữ được tri thức cũ mà còn phát hiện malware tốt hơn. Recall tăng rõ nhất, nghĩa là mô hình bỏ sót ít malware hơn.

### 4.2. Đánh giá theo catastrophic forgetting

Forget càng thấp càng tốt:

```text
FedAvg          = 0.0407
FL-MalDrift     = 0.0361
FL-CL-MalDrift  = 0.0108
```

FL-CL-MalDrift giảm Forget khoảng 73.5% so với FedAvg:

```text
(0.0407 - 0.0108) / 0.0407 ≈ 73.5%
```

So với FL-MalDrift, FL-CL-MalDrift giảm Forget khoảng 70.1%:

```text
(0.0361 - 0.0108) / 0.0361 ≈ 70.1%
```

Đây là bằng chứng mạnh nhất trong bảng cho thấy Replay Buffer và EWC giúp hạn chế quên task cũ.

### 4.3. Đánh giá theo BWT

BWT càng cao càng tốt. Nếu BWT âm, nghĩa là học task mới làm giảm hiệu năng trên task cũ.

```text
FedAvg          = -0.0382
FL-MalDrift     = -0.0321
FL-CL-MalDrift  = -0.0050
```

FedAvg và FL-MalDrift có BWT âm khá rõ. FL-CL-MalDrift vẫn hơi âm, nhưng gần 0 hơn nhiều. Điều này cho thấy khi học task mới, FL-CL-MalDrift ít làm hỏng hiệu năng trên task cũ hơn.

### 4.4. Đánh giá theo FWT

FWT thể hiện khả năng kiến thức từ task trước hỗ trợ task sau.

```text
FedAvg          = 0.3780
FL-MalDrift     = 0.3768
FL-CL-MalDrift  = 0.3812
```

FL-CL-MalDrift cao nhất, nhưng khoảng cách FWT không lớn. Vì vậy trong kết quả này, FWT chỉ nên dùng như chỉ số hỗ trợ. Hai chỉ số chứng minh rõ nhất cho phần continual learning là Forget và BWT.

## 5. Kết Luận Có Thể Đưa Vào Báo Cáo

Với cấu hình `category_dominant` và `dominant_ratio=0.90`, mỗi task vẫn chứa đủ các họ malware nhưng tỷ trọng từng họ thay đổi mạnh theo thời gian. Cách chia này tạo concept drift rõ hơn so với `dominant_ratio=0.75`, đồng thời vẫn hợp lý hơn `category_strict` vì không biến mỗi task thành một domain quá tách biệt.

Kết quả 5-fold cho thấy FL-CL-MalDrift đạt hiệu năng tốt nhất trên hầu hết các chỉ số. Cụ thể, FL-CL-MalDrift đạt ACC = 0.9089, F1 = 0.9320 và Recall = 0.9511, cao hơn FedAvg và FL-MalDrift. Đặc biệt, Forget của FL-CL-MalDrift chỉ còn 0.0108, thấp hơn nhiều so với FedAvg 0.0407 và FL-MalDrift 0.0361. BWT của FL-CL-MalDrift cũng gần 0 nhất, cho thấy mô hình ít bị suy giảm hiệu năng trên task cũ khi học task mới.

Do đó, kết quả mới nhất chứng minh rõ vai trò của Replay Buffer và EWC trong FL-CL-MalDrift: phương pháp này vừa cải thiện khả năng phát hiện malware, vừa giảm catastrophic forgetting trong môi trường federated continual learning có concept drift.


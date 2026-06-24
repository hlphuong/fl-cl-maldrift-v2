# Giải Thích Luồng Thực Nghiệm Và Kết Quả

Tài liệu này giải thích chi tiết luồng chạy, cách chia dữ liệu, ngưỡng xác định drift, ý nghĩa các chỉ số, và vì sao kết quả giữa dataset 1:1 và 2:1 khác nhau.

## 1. Mục Tiêu Thực Nghiệm

Project so sánh 3 phương pháp trong bài toán phát hiện malware Android dưới concept drift:

```text
FedAvg
  Federated Learning cơ bản.
  Không có drift-aware aggregation.
  Không có continual learning.

FL-MalDrift
  Có drift-aware aggregation.
  Server lọc hoặc giảm trọng số client có drift score cao.
  Không có Replay/EWC nên vẫn có nguy cơ quên task cũ.

FL-CL-MalDrift
  Kế thừa FL-MalDrift.
  Thêm Drift Resolution Controller, Replay Buffer và EWC.
  Mục tiêu là vừa phát hiện malware tốt, vừa giảm catastrophic forgetting.
```

## 2. Dataset CICMalDroid Được Xử Lý Như Thế Nào

CSV gốc được đọc từ thư mục `CSV/`, ưu tiên file:

```text
feature_vectors_syscallsbinders_frequency_5_Cat.csv
```

Nhãn gốc là `Class`:

```text
1 = Adware
2 = Banking
3 = SMS malware
4 = Riskware
5 = Benign
```

Khi chuyển sang bài toán binary malware detection:

```text
Class 1..4 -> label 1 = malware
Class 5    -> label 0 = benign
```

Các bước tiền xử lý:

```text
1. Đọc CSV gốc.
2. Chọn các cột numeric.
3. Xử lý NaN và Inf.
4. Loại cột constant.
5. Chọn top 200 features bằng mutual information.
6. Lấy mẫu theo tỷ lệ malware:benign.
7. Lưu ra data/cicmaldroid/features.csv.
```

## 3. Dataset 1:1 Và 2:1

### Dataset 1:1

Lệnh tạo:

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 1.0
```

Số lượng:

```text
Benign  = 1795
Malware = 1795
Total   = 3590
```

Với tỷ lệ 1:1, mô hình học benign và malware cân bằng. Kết quả detection có thể ổn, nhưng recall trên malware thường không được nhấn mạnh bằng tỷ lệ 2:1.

Kết quả single-run đã ghi nhận:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.7848 | 0.8304 | 0.7521 | 0.7235 | 0.0000 | 0.0406 | 0.1878 |
| FL-MalDrift | 0.7845 | 0.8288 | 0.7529 | 0.7269 | 0.0000 | 0.1123 | 0.1396 |
| FL-CL-MalDrift | 0.8303 | 0.8299 | 0.8162 | 0.8369 | 0.0000 | 0.1082 | 0.1847 |

Bang này cho thấy FL-CL-MalDrift cải thiện ACC/F1/Recall, nhưng `Forget = 0`, nên chưa phải bằng chứng mạnh cho phần catastrophic forgetting.

### Dataset 2:1

Lệnh tạo:

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 2.0
```

Số lượng hiện tại:

```text
Benign  = 1795
Malware = 3590
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

Với tỷ lệ 2:1, mô hình nhìn thấy nhiều malware hơn, nên thường học tốt hơn đặc trưng malware. Recall và F1 thường tăng. Tuy nhiên Precision có thể giảm nhẹ nếu mô hình dự đoán malware mạnh hơn và báo nhầm một số benign thành malware.

Kết quả 2:1, single-run trước khi đổi sang `category_dominant`:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.9163 | 0.9154 | 0.9385 | 0.9642 | 0.0107 | 0.0003 | 0.2631 |
| FL-MalDrift | 0.9163 | 0.9147 | 0.9386 | 0.9652 | 0.0095 | 0.0013 | 0.2655 |
| FL-CL-MalDrift | 0.9217 | 0.9147 | 0.9430 | 0.9738 | 0.0046 | 0.0107 | 0.3372 |

Kết quả 2:1, 5-fold, 5 clients trước khi đổi sang `category_dominant`:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.9033 ± 0.0145 | 0.9122 ± 0.0067 | 0.9283 ± 0.0116 | 0.9471 ± 0.0199 | 0.0206 ± 0.0069 | -0.0150 ± 0.0108 | 0.2506 ± 0.0256 |
| FL-MalDrift | 0.9026 ± 0.0149 | 0.9113 ± 0.0064 | 0.9279 ± 0.0120 | 0.9469 ± 0.0203 | 0.0226 ± 0.0092 | -0.0169 ± 0.0128 | 0.2524 ± 0.0204 |
| FL-CL-MalDrift | 0.9169 ± 0.0112 | 0.9143 ± 0.0122 | 0.9394 ± 0.0082 | 0.9669 ± 0.0112 | 0.0071 ± 0.0027 | 0.0033 ± 0.0062 | 0.3151 ± 0.0227 |

Kết luận từ bảng 5-fold:

```text
FL-CL-MalDrift có ACC, F1, Recall cao nhất.
FL-CL-MalDrift có Forget thấp nhất.
FL-CL-MalDrift có BWT dương, trong khi FedAvg và FL-MalDrift âm.
FL-CL-MalDrift có FWT cao nhất.
```

## 4. Dữ Liệu Được Chia Thành Task Như Thế Nào

Với `--task_strategy category_dominant`, dữ liệu được chia thành 5 task. Điểm quan trọng là task nào cũng có đủ 4 họ malware:

```text
Task 0 = đủ Adware, Banking, SMS, Riskware; Adware chiếm đa số
Task 1 = đủ Adware, Banking, SMS, Riskware; Banking chiếm đa số
Task 2 = đủ Adware, Banking, SMS, Riskware; SMS chiếm đa số
Task 3 = đủ Adware, Banking, SMS, Riskware; Riskware chiếm đa số
Task 4 = đủ Adware, Banking, SMS, Riskware; SMS chiếm đa số lần 2
```

Với dataset 2:1 hiện tại:

| Task | Họ chiếm đa số | Adware | Banking | SMS | Riskware | Benign | Total |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Adware | 359 | 49 | 117 | 58 | 291 | 874 |
| 1 | Banking | 30 | 585 | 117 | 58 | 395 | 1185 |
| 2 | SMS | 30 | 49 | 527 | 58 | 332 | 996 |
| 3 | Riskware | 30 | 49 | 117 | 693 | 445 | 1334 |
| 4 | SMS | 30 | 49 | 527 | 58 | 332 | 996 |

Nếu muốn làm rõ catastrophic forgetting hơn, dùng thêm:

```text
--dominant_ratio 0.90
```

Khi đó mỗi task vẫn có đủ 4 họ malware, nhưng họ chiếm đa số sẽ áp đảo hơn. Ví dụ Adware-dominant sẽ có rất nhiều Adware và chỉ còn ít Banking/SMS/Riskware. Khi mô hình học sang task Banking/SMS/Riskware, phân phối Adware cũ xuất hiện ít hơn, nên FedAvg và FL-MalDrift dễ giảm hiệu năng trên task cũ hơn. FL-CL-MalDrift có Replay Buffer và EWC nên có điều kiện thể hiện rõ hơn ở `Forget` thấp hơn và `BWT` ít âm hơn.

Trong mỗi task, code chia tiếp:

```text
70% train
10% validation
20% test
```

Sau đó train set của từng task được chia cho các client theo `--partition_strategy category` để tạo non-IID.

## 5. K-Fold Và Task Chạy Theo Thứ Tự Nào

Thứ tự đúng là:

```text
K-Fold là vòng ngoài.
Task là vòng trong.
```

Cụ thể:

```text
Fold 1:
  train subset -> chia thành Task 0..4 -> học tuần tự Task 0 -> 4

Fold 2:
  train subset -> chia thành Task 0..4 -> học tuần tự Task 0 -> 4

...

Fold 5:
  train subset -> chia thành Task 0..4 -> học tuần tự Task 0 -> 4
```

Với 5-fold trên 5385 mẫu:

```text
Mỗi fold:
  Outer train = 4308 mẫu = 2872 malware + 1436 benign
  Outer test  = 1077 mẫu = 718 malware + 359 benign
```

Trong code hiện tại, K-Fold dùng train subset để tạo 5 continual tasks. Các metric continual learning được tính trên test nội bộ của từng task.

## 6. Ngưỡng Drift Được Xác Định Như Thế Nào

### Drift score của client

Trong mỗi local round:

```text
1. Client train trên local data.
2. Tạo error stream: error = 1 nếu dự đoán sai, 0 nếu dự đoán đúng.
3. Drift detector cập nhật trên error stream.
4. Detector trả về drift_score trong khoảng [0,1].
```

`drift_score` càng cao thì client càng có dấu hiệu drift.

### Ngưỡng server `tau`

Server có ngưỡng drift `tau`.

Trong các lệnh thực nghiệm chính:

```text
tau_init = 0.25
tau_min  = 0.2
warmup_rounds = 0
```

Client được xem là drift nếu:

```text
drift_score > tau
```

Server drift-aware sẽ:

```text
accept client nếu drift_score <= 1.5 * tau
down-weight update theo w = n_samples * max(0.1, 1 - drift_score)
cập nhật tau bằng EWMA dựa trên các drift score gần đây
```

Công thức cập nhật `tau` trong `fl/server.py`:

```text
tau_hat = mean_recent_score + k_sigma * std_recent_score

tau_t = clip(
    ewma_alpha * tau_{t-1}
    + (1 - ewma_alpha) * tau_hat
    + eta * (target_participation - participation_t),
    tau_min,
    tau_max
)
```

Với config mặc định:

```text
ewma_alpha = 0.8
k_sigma = 1.5
target_participation = 0.7
eta = 0.05
tau_max = 1.0
```

### DRC stage

DRC dùng `drift_score` và `tau` để chọn stage:

```text
score <= tau:
  STABLE

score > tau, count <= K1:
  REPLAY

score > tau, K1 < count <= K2:
  EWC

score > tau, count > K2:
  ESCALATION, client bị withheld
```

Config mặc định:

```text
K1 = 3
K2 = 8
R = 3
delta = 0.05
ema_alpha = 0.3
```

Recovery:

```text
tau_re = tau + delta
nếu EMA(drift_score) < tau_re trong R round liên tiếp
-> client recovered và rejoin federation
```

`--drc_stress` sẽ hạ:

```text
K1 = 1
K2 = 1
```

Mục đích là ép DRC đi qua Escalation/Recovery để kiểm tra drift loop.

## 7. Ý Nghĩa Các Chỉ Số

| Metric | Ý nghĩa | Tốt khi |
|---|---|---|
| ACC | Tỷ lệ dự đoán đúng trên tổng mẫu | Cao |
| Precision | Trong các mẫu bị báo là malware, bao nhiêu thật sự là malware | Cao |
| Recall | Trong malware thật, mô hình bắt được bao nhiêu | Cao |
| F1 | Trung bình điều hòa giữa Precision và Recall | Cao |
| Forget | Mức giảm hiệu năng trên task cũ sau khi học task mới | Thấp |
| BWT | Ảnh hưởng của task mới lên task cũ | Cao, gần 0 hoặc dương |
| FWT | Kiến thức task cũ giúp task mới trước khi học task mới | Cao |
| EscRate | Tỷ lệ drift dẫn đến escalation | > 0 trong stress test |
| RecRate | Tỷ lệ client escalation có thể recovery | Cao trong stress test |

## 8. Vì Sao 2:1 Cho Kết Quả Khác 1:1

### 1. Mô hình thấy nhiều malware hơn

Với 2:1, malware gấp đôi benign. Mô hình học được nhiều pattern malware hơn:

```text
system call frequency
binder call frequency
hành vi bất thường của từng họ malware
```

Vì vậy Recall và F1 thường tăng.

### 2. Recall tăng nhưng Precision có thể dao động

Khi malware nhiều hơn, mô hình có xu hướng dự đoán malware mạnh hơn:

```text
Recall tăng:
  Ít bỏ sót malware hơn.

Precision có thể giảm nhẹ:
  Có thể có thêm benign bị báo nhầm là malware.
```

Trong malware detection, Recall cao rất quan trọng vì bỏ sót malware nguy hiểm hơn báo nhầm benign.

### 3. `category_dominant` tạo drift hợp lý hơn

`category_dominant` tạo chuỗi task:

```text
Adware-dominant -> Banking-dominant -> SMS-dominant -> Riskware-dominant -> SMS-dominant
```

Mỗi task vẫn có đủ Adware, Banking, SMS và Riskware, nên mô hình không bị rơi vào tình huống quá nhân tạo là task chỉ chứa một họ malware. Concept drift đến từ việc tỷ trọng từng họ malware thay đổi theo thời gian. Cách này hợp lý hơn để chứng minh catastrophic forgetting: mô hình vẫn luôn thấy các họ malware cũ, nhưng nếu không có Replay/EWC thì hiệu năng trên các phân phối cũ vẫn có thể giảm khi task mới chiếm ưu thế.

### 4. Vì sao FL-MalDrift đôi khi gần FedAvg

FL-MalDrift chủ yếu xử lý drift bằng filtering/down-weighting. Nếu drift score không quá cao hoặc phần lớn client vẫn được accept, kết quả có thể gần FedAvg.

FL-CL-MalDrift khác ở chỗ có thêm:

```text
Replay Buffer:
  Học lại một phần mẫu cũ.

EWC:
  Giữ các trọng số quan trọng của task cũ.

DRC:
  Chuyển stage khi drift kéo dài.
```

Vì vậy FL-CL-MalDrift thường tốt hơn ở Forget, BWT và FWT.

## 9. Diễn Giải Kết Quả 5-Fold Hiện Tại

Kết quả 5-fold, 5 clients:

```text
FedAvg:
  ACC    = 0.9033 ± 0.0145
  F1     = 0.9283 ± 0.0116
  Recall = 0.9471 ± 0.0199
  Forget = 0.0206 ± 0.0069
  BWT    = -0.0150 ± 0.0108
  FWT    = 0.2506 ± 0.0256

FL-MalDrift:
  ACC    = 0.9026 ± 0.0149
  F1     = 0.9279 ± 0.0120
  Recall = 0.9469 ± 0.0203
  Forget = 0.0226 ± 0.0092
  BWT    = -0.0169 ± 0.0128
  FWT    = 0.2524 ± 0.0204

FL-CL-MalDrift:
  ACC    = 0.9169 ± 0.0112
  F1     = 0.9394 ± 0.0082
  Recall = 0.9669 ± 0.0112
  Forget = 0.0071 ± 0.0027
  BWT    = 0.0033 ± 0.0062
  FWT    = 0.3151 ± 0.0227
```

Kết luận:

```text
FL-CL-MalDrift có khả năng phát hiện malware tốt hơn.
FL-CL-MalDrift giảm forgetting rõ rệt.
FL-CL-MalDrift có BWT dương, chứng tỏ học task mới không làm suy giảm task cũ như FedAvg/FL-MalDrift.
FL-CL-MalDrift có FWT cao nhất, chứng tỏ kiến thức cũ chuyển giao sang task mới tốt hơn.
```

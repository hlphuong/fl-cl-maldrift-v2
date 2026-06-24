# Giai thich luong thuc nghiem va ket qua

File nay giai thich chi tiet luong chay, cach chia du lieu, nguong drift, cac chi so danh gia, va vi sao ket qua 1:1 khac 2:1.

## 1. Muc tieu thuc nghiem

Project so sanh 3 phuong phap trong bai toan phat hien malware Android co concept drift:

```text
FedAvg
  Federated Learning co ban, khong drift-aware, khong continual learning.

FL-MalDrift
  Co drift-aware aggregation: loc/down-weight client co drift score cao.
  Khong co replay/EWC nen van co nguy co quen task cu.

FL-CL-MalDrift
  FL-MalDrift + Drift Resolution Controller + Replay Buffer + EWC.
  Muc tieu: vua phat hien malware tot, vua giam catastrophic forgetting.
```

## 2. Du lieu CICMalDroid duoc xu ly the nao

CSV raw duoc doc tu thu muc `CSV/`, uu tien file:

```text
feature_vectors_syscallsbinders_frequency_5_Cat.csv
```

Cot label goc la `Class`:

```text
1 = Adware
2 = Banking
3 = SMS malware
4 = Riskware
5 = Benign
```

Trong bai toan binary malware detection:

```text
Class 1..4 -> label 1 = malware
Class 5    -> label 0 = benign
```

Tien xu ly:

```text
1. Doc CSV raw
2. Lay cac cot numeric
3. Xu ly NaN/Inf
4. Loai cot constant
5. Chon top 200 features bang mutual information
6. Lay mau theo ty le malware:benign
7. Luu ra data/cicmaldroid/features.csv
```

## 3. Dataset 1:1 va 2:1

### Dataset 1:1

Lenh:

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 1.0
```

So luong:

```text
Benign  = 1795
Malware = 1795
Total   = 3590
```

Voi thiet lap 1:1, mo hinh hoc benign va malware can bang. Ket qua detection co the on, nhung recall malware thuong khong duoc nhan manh bang 2:1.

Ket qua single-run da ghi nhan:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.7848 | 0.8304 | 0.7521 | 0.7235 | 0.0000 | 0.0406 | 0.1878 |
| FL-MalDrift | 0.7845 | 0.8288 | 0.7529 | 0.7269 | 0.0000 | 0.1123 | 0.1396 |
| FL-CL-MalDrift | 0.8303 | 0.8299 | 0.8162 | 0.8369 | 0.0000 | 0.1082 | 0.1847 |

Bang nay cho thay FL-CL-MalDrift cai thien ACC/F1/Recall, nhung `Forget=0` nen chua phai bang chung manh ve catastrophic forgetting.

### Dataset 2:1

Lenh:

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 2.0
```

So luong hien tai:

```text
Benign  = 1795
Malware = 3590
Total   = 5385
```

Phan phoi class:

| Class | Loai | So mau |
|---:|---|---:|
| 1 | Adware | 479 |
| 2 | Banking | 781 |
| 3 | SMS | 1405 |
| 4 | Riskware | 925 |
| 5 | Benign | 1795 |

Voi 2:1, mo hinh thay nhieu malware hon, nen Recall/F1 tren malware thuong tang. Tuy nhien Precision co the giam nhe neu mo hinh du doan malware "manh tay" hon va bao nham mot so benign thanh malware.

Ket qua 2:1, `category_strict`, 10 clients, single-run:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.9163 | 0.9154 | 0.9385 | 0.9642 | 0.0107 | 0.0003 | 0.2631 |
| FL-MalDrift | 0.9163 | 0.9147 | 0.9386 | 0.9652 | 0.0095 | 0.0013 | 0.2655 |
| FL-CL-MalDrift | 0.9217 | 0.9147 | 0.9430 | 0.9738 | 0.0046 | 0.0107 | 0.3372 |

Ket qua 2:1, 5-fold, 5 clients:

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.9033 +/- 0.0145 | 0.9122 +/- 0.0067 | 0.9283 +/- 0.0116 | 0.9471 +/- 0.0199 | 0.0206 +/- 0.0069 | -0.0150 +/- 0.0108 | 0.2506 +/- 0.0256 |
| FL-MalDrift | 0.9026 +/- 0.0149 | 0.9113 +/- 0.0064 | 0.9279 +/- 0.0120 | 0.9469 +/- 0.0203 | 0.0226 +/- 0.0092 | -0.0169 +/- 0.0128 | 0.2524 +/- 0.0204 |
| FL-CL-MalDrift | 0.9169 +/- 0.0112 | 0.9143 +/- 0.0122 | 0.9394 +/- 0.0082 | 0.9669 +/- 0.0112 | 0.0071 +/- 0.0027 | 0.0033 +/- 0.0062 | 0.3151 +/- 0.0227 |

Ket luan tu bang 5-fold:

```text
FL-CL-MalDrift co ACC, F1, Recall cao nhat.
FL-CL-MalDrift co Forget thap nhat.
FL-CL-MalDrift co BWT duong, trong khi FedAvg va FL-MalDrift am.
FL-CL-MalDrift co FWT cao nhat.
```

## 4. Chia task nhu the nao

Voi `--task_strategy category_strict`, du lieu duoc chia thanh 5 task:

```text
Task 0 = Adware-dominant
Task 1 = Banking-dominant
Task 2 = SMS-dominant
Task 3 = Riskware-dominant
Task 4 = MixedHeldout
```

Voi dataset 2:1 hien tai, toan bo 5 task co:

| Task | Domain | Malware | Benign | Total |
|---:|---|---:|---:|---:|
| 0 | Adware | 408 | 204 | 612 |
| 1 | Banking | 664 | 332 | 996 |
| 2 | SMS | 1195 | 598 | 1793 |
| 3 | Riskware | 787 | 393 | 1180 |
| 4 | MixedHeldout | 536 | 268 | 804 |

Trong moi task, code chia tiep:

```text
70% train
10% validation
20% test
```

Sau do train set cua moi task duoc chia cho client theo `--partition_strategy category` de tao non-IID.

## 5. K-Fold va Task chay theo thu tu nao

Thu tu dung la:

```text
K-Fold la vong ngoai.
Task la vong trong.
```

Cu the:

```text
Fold 1:
  train subset -> chia thanh Task 0..4 -> hoc tuan tu Task 0 -> 4

Fold 2:
  train subset -> chia thanh Task 0..4 -> hoc tuan tu Task 0 -> 4

...

Fold 5:
  train subset -> chia thanh Task 0..4 -> hoc tuan tu Task 0 -> 4
```

Voi 5-fold tren 5385 mau:

```text
Moi fold:
  Outer train = 4308 mau = 2872 malware + 1436 benign
  Outer test  = 1077 mau = 718 malware + 359 benign
```

Luu y: Trong code hien tai, K-Fold dung train subset de tao 5 continual tasks. Cac metric continual learning duoc tinh tren test noi bo cua tung task.

## 6. Nguong drift duoc xac dinh the nao

### Drift score cua client

Trong moi local round:

```text
1. Client train tren local data.
2. Lay stream loi du doan: error = 1 neu du doan sai, 0 neu dung.
3. Drift detector cap nhat tren error stream.
4. Detector tra ve drift_score trong [0,1].
```

`drift_score` cang cao thi client cang co dau hieu drift.

### Nguong server tau

Server co nguong drift `tau`.

Trong cac lenh thuc nghiem chinh:

```text
tau_init = 0.25
tau_min  = 0.2
warmup_rounds = 0
```

Client duoc xem la drift neu:

```text
drift_score > tau
```

Server drift-aware se:

```text
accept client neu drift_score <= 1.5 * tau
down-weight update theo w = n_samples * max(0.1, 1 - drift_score)
cap nhat tau bang EWMA dua tren cac drift score gan day
```

Cong thuc tau trong `fl/server.py`:

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

Voi config mac dinh:

```text
ewma_alpha = 0.8
k_sigma = 1.5
target_participation = 0.7
eta = 0.05
tau_max = 1.0
```

### DRC stage

DRC dung `drift_score` va `tau` de chon stage:

```text
score <= tau:
  STABLE

score > tau, count <= K1:
  REPLAY

score > tau, K1 < count <= K2:
  EWC

score > tau, count > K2:
  ESCALATION, client bi withheld
```

Config mac dinh:

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
neu EMA(drift_score) < tau_re trong R round lien tiep
-> client recovered va rejoin federation
```

`--drc_stress` se ha:

```text
K1 = 1
K2 = 1
```

Muc dich la ep DRC di qua Escalation/Recovery de kiem tra drift loop.

## 7. Cac chi so ket qua

| Metric | Y nghia | Tot khi |
|---|---|---|
| ACC | Ty le du doan dung tren tong mau | Cao |
| Precision | Trong cac mau bi bao la malware, bao nhieu that su la malware | Cao |
| Recall | Trong malware that, mo hinh bat duoc bao nhieu | Cao |
| F1 | Trung binh dieu hoa Precision va Recall | Cao |
| Forget | Muc giam hieu nang tren task cu sau khi hoc task moi | Thap |
| BWT | Anh huong cua task moi len task cu | Cao, gan 0 hoac duong |
| FWT | Kien thuc task cu giup task moi truoc khi hoc task moi | Cao |
| EscRate | Ty le drift dan den escalation | >0 trong stress test |
| RecRate | Ty le client escalation co the recovery | Cao trong stress test |

## 8. Vi sao 2:1 cho ket qua khac 1:1

### 1. Mo hinh thay nhieu malware hon

Voi 2:1, malware gap doi benign. Mo hinh hoc duoc nhieu pattern malware hon:

```text
system call frequency
binder call frequency
hanh vi bat thuong cua tung ho malware
```

Vi vay Recall va F1 thuong tang.

### 2. Recall tang nhung Precision co the dao dong

Khi malware nhieu hon, mo hinh co xu huong du doan malware manh hon:

```text
Tang Recall: bot bo sot malware
Co the giam Precision nhe: co them benign bi bao nham la malware
```

Trong malware detection, Recall cao rat quan trong vi bo sot malware nguy hiem hon bao nham benign.

### 3. category_strict tao drift ro hon

Neu dung `category_strict`, moi task co mot domain malware ro:

```text
Adware -> Banking -> SMS -> Riskware -> MixedHeldout
```

Vi task thay doi ro rang, catastrophic forgetting de quan sat hon. Replay + EWC cua FL-CL-MalDrift co co hoi the hien tac dung.

### 4. Vi sao FL-MalDrift doi khi gan FedAvg

FL-MalDrift chu yeu xu ly update drift bang filtering/down-weighting. Neu drift score khong qua cao hoac client van con duoc accept nhieu, ket qua co the gan FedAvg.

FL-CL-MalDrift khac o cho co:

```text
Replay Buffer: hoc lai mot phan mau cu
EWC: giu cac trong so quan trong cua task cu
DRC: chuyen stage khi drift keo dai
```

Nen no thuong tot hon ve Forget/BWT/FWT.

## 9. Dien giai ket qua 5-fold hien tai

Bang 5-fold, 5 clients:

```text
FedAvg:
  ACC    = 0.9033 +/- 0.0145
  F1     = 0.9283 +/- 0.0116
  Recall = 0.9471 +/- 0.0199
  Forget = 0.0206 +/- 0.0069
  BWT    = -0.0150 +/- 0.0108
  FWT    = 0.2506 +/- 0.0256

FL-MalDrift:
  ACC    = 0.9026 +/- 0.0149
  F1     = 0.9279 +/- 0.0120
  Recall = 0.9469 +/- 0.0203
  Forget = 0.0226 +/- 0.0092
  BWT    = -0.0169 +/- 0.0128
  FWT    = 0.2524 +/- 0.0204

FL-CL-MalDrift:
  ACC    = 0.9169 +/- 0.0112
  F1     = 0.9394 +/- 0.0082
  Recall = 0.9669 +/- 0.0112
  Forget = 0.0071 +/- 0.0027
  BWT    = 0.0033 +/- 0.0062
  FWT    = 0.3151 +/- 0.0227
```

Ket luan:

```text
FL-CL-MalDrift co kha nang phat hien tot hon.
FL-CL-MalDrift giam forgetting ro ret.
FL-CL-MalDrift co BWT duong, chung to hoc task moi khong lam suy giam task cu nhu FedAvg/FL-MalDrift.
FL-CL-MalDrift co FWT cao nhat, chung to kien thuc cu chuyen giao sang task moi tot hon.
```


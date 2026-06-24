# Huong dan chay FL-CL-MalDrift v2

File nay huong dan cach cai dat, chuan bi du lieu, chay so sanh 3 phuong phap, chay K-Fold, xuat bang ket qua, va vai tro cac file chinh trong project.

## 1. Cau truc thu muc

```text
FL_CL_MalDrift_v2/
|-- CSV/                                  # CSV raw tu CICMalDroid
|-- FL_CL_MalDrift_Summary_1.docx         # file tom tat/yeu cau thiet ke
|-- fl_cl_maldrift_v2/
|   |-- main.py                           # entry point chay training/testing
|   |-- config.py                         # cau hinh hyperparameter tap trung
|   |-- prepare_cicmaldroid.py            # tien xu ly CICMalDroid -> features.csv
|   |-- requirements.txt                  # cac thu vien Python can cai
|   |-- data/
|   |   |-- dataset.py                    # load data, chia task, chia client
|   |   |-- cicmaldroid/features.csv      # dataset da tien xu ly
|   |-- fl/
|   |   |-- client.py                     # logic client FL + train local + DRC
|   |   |-- server.py                     # FedAvg/drift-aware aggregation
|   |-- models/mlp.py                     # mo hinh MalwareMLP
|   |-- modules/
|   |   |-- drift_detector.py             # HDDM-W/ADWIN/DDM/EDDM drift detector
|   |   |-- drc.py                        # Drift Resolution Controller
|   |   |-- continual_learning.py         # Replay Buffer + EWC
|   |   |-- privacy.py                    # DP-SGD clipping/noise
|   |-- utils/
|   |   |-- metrics.py                    # ACC/F1/Recall/Forget/BWT/FWT
|   |   |-- logger.py                     # luu json/csv/plot
|   |-- scripts/
|   |   |-- export_kfold_task_table.py    # xuat bang CSV/PNG tong hop K-Fold
|   |-- results/                          # ket qua thuc nghiem
```

## 2. Cai dat moi truong

Tu thu muc goc:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2
```

Neu da co moi truong `malenv`:

```powershell
.\malenv\Scripts\python.exe --version
```

Neu clone lai tu GitHub va chua co moi truong:

```powershell
python -m venv malenv
.\malenv\Scripts\python.exe -m pip install --upgrade pip
.\malenv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Chuan bi dataset

Script `prepare_cicmaldroid.py` doc CSV raw tu thu muc `..\CSV`, chon 200 feature tot nhat, encode label, va ghi ra:

```text
fl_cl_maldrift_v2/data/cicmaldroid/features.csv
```

### Tao dataset 1:1

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 1.0
```

Ket qua du kien:

```text
Benign  = 1795
Malware = 1795
Total   = 3590
```

### Tao dataset 2:1

```powershell
.\malenv\Scripts\python.exe prepare_cicmaldroid.py --force --malware_ratio 2.0
```

Ket qua hien tai:

```text
Benign  = 1795
Malware = 3590
Total   = 5385
Class dist:
  1 Adware   = 479
  2 Banking  = 781
  3 SMS      = 1405
  4 Riskware = 925
  5 Benign   = 1795
```

## 4. Chay so sanh 3 phuong phap

3 phuong phap:

```text
FedAvg:          Federated Learning thuong
FL-MalDrift:     Drift-aware aggregation, khong Continual Learning
FL-CL-MalDrift:  Drift-aware + DRC + Replay + EWC
```

Lenh so sanh chinh voi 5 task, 5 client:

```powershell
.\malenv\Scripts\python.exe main.py --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_strict --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --out results\compare_5clients_strict_2to1
```

Y nghia cac tham so quan trong:

```text
--compare                    chay ca 3 method
--dataset cicmaldroid        dung CICMalDroid da tien xu ly
--tasks 5                    chia du lieu thanh 5 task
--rounds 25                  moi task co 25 communication rounds
--clients 5                  dung 5 federated clients
--task_strategy category_strict
                             task 0..3 gan voi tung ho malware, task 4 mixed/held-out
--partition_strategy category
                             chia train data cho client theo non-IID category
--local_epochs 3             moi client train 3 epoch local moi round
--tau_init 0.25              nguong drift ban dau
--tau_min 0.2                san nguong drift
--warmup_rounds 0            bat dau loc drift ngay tu round dau
--ewc_lambda 0.1             do manh EWC
--replay_buffer 800          kich thuoc replay buffer
```

## 5. Chay K-Fold 5 folds, 5 clients, 3 method

Day la lenh nen dung de lay bang ket qua on dinh nhat:

```powershell
.\malenv\Scripts\python.exe main.py --kfold 5 --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_strict --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --out results\kfold5_clients5_compare
```

Luong chay:

```text
Fold 1 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4 -> 3 methods
Fold 2 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4 -> 3 methods
Fold 3 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4 -> 3 methods
Fold 4 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4 -> 3 methods
Fold 5 -> Task 0 -> Task 1 -> Task 2 -> Task 3 -> Task 4 -> 3 methods
```

File ket qua quan trong:

```text
results/kfold5_clients5_compare/kfold_per_fold.csv
results/kfold5_clients5_compare/kfold_per_task.csv
results/kfold5_clients5_compare/kfold_summary.csv
```

## 6. Xuat bang tong hop CSV/PNG

Sau khi chay K-Fold, xuat bang tong hop theo format Scenario/Task/Method:

```powershell
.\malenv\Scripts\python.exe scripts\export_kfold_task_table.py --result_dir results\kfold5_clients5_compare --out_prefix kfold_task_summary_table_full
```

File xuat ra:

```text
results/kfold5_clients5_compare/kfold_task_summary_table_full.csv
results/kfold5_clients5_compare/kfold_task_summary_table_full.png
```

Trong bang:

```text
Scenario = Fold 1..Fold 5 va Mean +/- Std
Task     = Task 0..Task 4, Overall
Method   = FedAvg, FL-MalDrift, FL-CL-MalDrift
Metric   = Acc, Prec, Rec, F1, Forget, BWT, FWT
```

Luu y: `Forget`, `BWT`, `FWT` la metric cua ca chuoi continual learning, nen chi co o dong `Overall`, khong co rieng cho tung task.

## 7. Chay DRC stress test

Dung de kiem tra drift loop/escalation/recovery:

```powershell
.\malenv\Scripts\python.exe main.py --method fl_cl_maldrift --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --task_strategy category_strict --partition_strategy category --local_epochs 3 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.1 --replay_buffer 800 --drc_stress --out results\drc_stress_5clients_2to1
```

Khi doc ket qua:

```text
EscRate > 0  -> co escalation that su xay ra
RecRate > 0  -> client co the recovery va quay lai federation
```

## 8. Vai tro cac file chinh

| File | Vai tro |
|---|---|
| `main.py` | Parse CLI, tao config, chay task, chay compare, chay K-Fold, in summary. |
| `config.py` | Noi tap trung hyperparameter: FL, DATA, DRIFT, DRC, CL, SERVER. |
| `prepare_cicmaldroid.py` | Chuyen CSV raw thanh `features.csv`, ho tro `--malware_ratio`. |
| `data/dataset.py` | Load dataset, chia task (`category`, `category_strict`, `category_revisit`), chia client (`dirichlet`, `category`). |
| `models/mlp.py` | Dinh nghia MalwareMLP. |
| `fl/client.py` | Client FL: nhan global model, train local, tinh drift score, goi DRC, tra update. |
| `fl/server.py` | Server FL: FedAvg, drift-aware filtering, drift-aware weighting, cap nhat tau. |
| `modules/drift_detector.py` | Drift detector sinh drift score trong [0,1]. |
| `modules/drc.py` | Drift Resolution Controller: STABLE, REPLAY, EWC, ESCALATION, RECOVERY. |
| `modules/continual_learning.py` | Replay Buffer va EWC. |
| `utils/metrics.py` | Tinh ACC, Precision, Recall, F1, Forgetting, BWT, FWT. |
| `utils/logger.py` | Luu JSON, CSV, PNG ket qua. |
| `scripts/export_kfold_task_table.py` | Xuat bang tong hop K-Fold thanh CSV va PNG. |


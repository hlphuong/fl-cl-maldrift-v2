# Handoff cho Claude - FL-CL-MalDrift

File này dùng để chuyển toàn bộ ngữ cảnh làm việc từ Codex sang Claude. Nhiệm vụ tiếp theo của Claude: **viết báo cáo/bài báo bằng tiếng Anh theo đúng mẫu/template mà người dùng đã cung cấp cho Claude**, dựa trên code, báo cáo, slide và kết quả thực nghiệm hiện có.

## 1. Mục tiêu đề tài

Tên đề tài:

**FL-CL-MalDrift: Học liên tục liên kết cho phát hiện mã độc Android thích nghi dưới concept drift**

Ý tưởng chính:

- Dựa trên hướng FL-MalDrift gốc cho phát hiện mã độc Android trong Federated Learning dưới concept drift.
- Mở rộng bằng Continual Learning để giảm **catastrophic forgetting** khi mô hình học qua nhiều task liên tiếp.
- Bổ sung Drift Resolution Controller (DRC), Replay Buffer, EWC và Recovery Monitor để mô hình vừa thích nghi với malware mới, vừa giữ tri thức trên malware cũ.

Ba phương pháp được so sánh:

| Phương pháp | Thành phần chính | Vai trò |
|---|---|---|
| FedAvg | Weighted averaging theo số mẫu | Baseline FL, không xử lý drift, không chống quên |
| FL-MalDrift | Drift-aware filtering/weighting | Giảm ảnh hưởng client có drift score cao |
| FL-CL-MalDrift | DRC + Replay Buffer + EWC + Recovery Monitor | Xử lý drift và giảm forgetting |

## 2. Các file quan trọng trong project

Thư mục gốc:

`C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2`

Các file/tài liệu quan trọng:

- `BAO_CAO_FL_CL_MalDrift.docx`: báo cáo Word đã viết.
- `BAO_CAO_KET_QUA_FL_CL_MalDrift.pptx`: slide báo cáo kết quả trước thầy.
- `pipeline_fl_cl_maldrift.png`: hình pipeline chương trình.
- `RUN_GUIDE.md`: hướng dẫn chạy lệnh.
- `EXPERIMENT_EXPLANATION.md`: giải thích luồng chạy, dataset, metric.
- `RESULT_NOTES.md`: nhận xét kết quả.
- `fl_cl_maldrift_v2/main.py`: entry point chạy thí nghiệm.
- `fl_cl_maldrift_v2/data/dataset.py`: xử lý dataset, chia task/client.
- `fl_cl_maldrift_v2/fl/client.py`: logic client, train local, Replay/EWC.
- `fl_cl_maldrift_v2/fl/server.py`: aggregation, drift-aware logic.
- `fl_cl_maldrift_v2/results/`: thư mục kết quả thực nghiệm.

Slide cũ định hướng ban đầu:

`C:\Users\phuong\Downloads\FL_CL_MalDrift_Slides_1.pptx`

## 3. Bộ dữ liệu sử dụng

Dataset:

**CICMalDroid 2020**

Các nhóm gốc:

- Adware
- Banking
- SMS
- Riskware
- Benign

Trong code, bài toán được chuyển thành binary malware detection:

- Class 1-4: Malware
- Class 5: Benign

Tiền xử lý:

- Đọc file feature vector của CICMalDroid.
- Chọn các cột numeric.
- Xử lý NaN/Inf.
- Loại cột constant.
- Chọn top 200 feature bằng mutual information.

Phân phối dữ liệu sau tiền xử lý với tỷ lệ malware:benign = 2:1:

| Class | Nhóm | Vai trò binary | Số mẫu |
|---|---|---|---:|
| 1 | Adware | Malware | 479 |
| 2 | Banking | Malware | 781 |
| 3 | SMS | Malware | 1405 |
| 4 | Riskware | Malware | 925 |
| 5 | Benign | Benign | 1795 |

Hai cấu hình dữ liệu đã chạy:

| Tỷ lệ | Số malware | Số benign | Tổng |
|---|---:|---:|---:|
| 1:1 | 1795 | 1795 | 3590 |
| 2:1 | 3590 | 1795 | 5385 |

Nhận xét:

- Tỷ lệ 1:1 cân bằng hơn nhưng số mẫu malware ít hơn, nên Recall/F1 thường thấp hơn và forgetting khó bật rõ.
- Tỷ lệ 2:1 giúp mô hình nhìn thấy nhiều malware hơn, nên Recall/F1 cao hơn.
- Khi tăng malware, Precision có thể không tăng tương ứng, vì mô hình nhạy hơn với malware và có thể dự đoán nhầm một số benign thành malware.

## 4. Cách chia fold, task và client

Thiết lập chính:

- K-Fold = 5.
- Mỗi fold là một lần thực nghiệm độc lập.
- Trong mỗi fold, phần train được chia thành 5 task tuần tự.
- Trong mỗi task, dữ liệu được chia cho 5 client.
- Client partition dùng Non-IID, thường với `alpha = 0.1`.

Luồng đúng:

```text
Dataset
  -> Stratified K-Fold, K = 5
      -> mỗi fold độc lập
          -> train split của fold được chia thành 5 continual tasks
              -> mỗi task được chia cho 5 clients
                  -> chạy FedAvg / FL-MalDrift / FL-CL-MalDrift
```

Điểm đã chỉnh quan trọng:

Ban đầu có ý tưởng mỗi task tương ứng một họ malware, nhưng cách này không hợp lý vì task sẽ quá tách biệt. Cách hiện tại đúng hơn:

- Mỗi task đều có đủ 4 họ malware: Adware, Banking, SMS, Riskware.
- Tuy nhiên mỗi task có một họ malware chiếm đa số để tạo concept drift có kiểm soát.
- Ví dụ:
  - Task 0: đủ 4 họ malware, Adware chiếm đa số.
  - Task 1: đủ 4 họ malware, Banking chiếm đa số.
  - Task 2: đủ 4 họ malware, SMS chiếm đa số.
  - Task 3: đủ 4 họ malware, Riskware chiếm đa số.
  - Task 4: mixed/held-out.

Tham số quan trọng:

- `--task_strategy category_dominant`
- `--dominant_ratio 0.75`: drift vừa, kết quả các phương pháp khá gần nhau.
- `--dominant_ratio 0.90`: drift rõ hơn, làm bật forgetting và lợi thế của FL-CL-MalDrift.

## 5. Các thành phần cải tiến của FL-CL-MalDrift

### DRC - Drift Resolution Controller

DRC chia phản ứng với drift thành các giai đoạn:

```text
Stable -> Replay -> EWC -> Escalation -> Recovery
```

Ý nghĩa:

- Stable: client hoạt động bình thường.
- Replay: dùng replay buffer để nhắc lại mẫu cũ khi drift bắt đầu xuất hiện.
- EWC: tăng ràng buộc trọng số quan trọng của task cũ.
- Escalation: nếu drift kéo dài, client có thể bị giảm ảnh hưởng hoặc withheld.
- Recovery: nếu drift score ổn định lại, client được đưa lại vào federation.

### Replay Buffer

- Lưu một phần mẫu cũ bằng reservoir sampling.
- Khi học task mới, replay mẫu cũ để giảm quên tri thức cũ.

### EWC

- Elastic Weight Consolidation.
- Bảo vệ các trọng số quan trọng đã học ở task trước.
- Phạt thay đổi mạnh các trọng số có Fisher diagonal lớn.

Ý tưởng công thức:

```text
L_total = L_task + lambda_EWC / 2 * sum_i F_i * (theta_i - theta_i_old)^2
```

### Recovery Monitor

- Theo dõi drift score bằng EMA.
- Nếu client ổn định đủ số round, cho client quay lại federation.

## 6. Ngưỡng drift và tham số

Ngưỡng drift `tau` trong code là ngưỡng lai:

- `tau_init` và `tau_min` được đặt từ lệnh chạy.
- Sau đó `tau` được cập nhật theo thống kê drift score và làm mượt bằng EWMA.

Vì vậy:

- `tau` không hoàn toàn cố định.
- Nhưng cũng không phải tham số model học tự do bằng backpropagation.
- Có thể nói: **ngưỡng được khởi tạo/cấu hình trước, sau đó tự động điều chỉnh theo phân phối drift score quan sát được**.

Các tham số chính:

| Tham số | Giá trị hay dùng | Ý nghĩa |
|---|---:|---|
| K1 | 3 | Số round drift liên tiếp để kích hoạt Replay |
| K2 | 8 | Nếu drift kéo dài, chuyển sang Escalation |
| R | 3 | Số round ổn định để Recovery |
| K-Fold | 5 | Số fold đánh giá |
| clients | 5 | Số client trong thực nghiệm chính |
| tasks | 5 | Số task continual |
| dominant_ratio | 0.90 | Tạo drift mạnh hơn giữa các task |
| replay_buffer | 500 | Kích thước replay buffer |
| ewc_lambda | 0.05 | Mức ràng buộc EWC |

## 7. Metric cần giải thích trong báo cáo

| Metric | Ý nghĩa | Cách diễn giải |
|---|---|---|
| ACC | Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| Precision | Độ chính xác khi dự đoán malware | Cao nghĩa là ít báo nhầm benign thành malware |
| Recall | Khả năng bắt malware | Cao nghĩa là ít bỏ sót malware |
| F1 | Trung hòa Precision và Recall | Quan trọng khi dữ liệu lệch lớp |
| Forgetting | Mức quên task cũ | Càng thấp càng tốt |
| BWT | Backward Transfer | Gần 0 hoặc dương càng tốt; âm nghĩa là học task mới làm giảm task cũ |
| FWT | Forward Transfer | Cao hơn nghĩa là tri thức trước hỗ trợ task mới tốt hơn |
| EscRate | Escalation Rate | Tần suất client bị đẩy sang escalation |
| RecRate | Recovery Rate | Tần suất client phục hồi |

Diễn giải Forgetting:

```text
Forgetting của task i = hiệu năng tốt nhất từng đạt trên task i - hiệu năng cuối cùng trên task i
```

Nếu Forgetting thấp:

- Mô hình ít quên malware cũ.
- FL-CL-MalDrift có Replay/EWC nên nên thấp hơn FedAvg/FL-MalDrift.

Diễn giải BWT:

```text
BWT đo tác động của việc học task mới lên task cũ.
```

- BWT âm mạnh: học task mới làm hỏng task cũ.
- BWT gần 0: ít ảnh hưởng xấu.
- BWT dương: học task mới còn giúp task cũ.

## 8. Lệnh chạy quan trọng

Chạy từ thư mục:

```powershell
cd C:\Users\phuong\Desktop\HKII\Malware\FL_CL_MalDrift_v2\fl_cl_maldrift_v2
```

### So sánh 3 phương pháp, single-run

```powershell
.\malenv\Scripts\python.exe main.py --compare --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --alpha 0.1 --local_epochs 3 --task_strategy category_dominant --dominant_ratio 0.90 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.05 --replay_buffer 500 --out results\compare_forgetting_strict_2to1
```

### So sánh 5 fold, 5 client, 3 phương pháp

```powershell
.\malenv\Scripts\python.exe main.py --compare --kfold 5 --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --alpha 0.1 --local_epochs 3 --task_strategy category_dominant --dominant_ratio 0.90 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.05 --replay_buffer 500 --out results\kfold5_clients5_dominant_r090
```

Thư mục kết quả chính:

`fl_cl_maldrift_v2\results\kfold5_clients5_dominant_r090`

Các file kết quả trong thư mục:

- `kfold_summary.csv`: bảng trung bình và độ lệch chuẩn qua 5 fold.
- `kfold_per_fold.csv`: kết quả từng fold.
- `kfold_per_task.csv`: kết quả từng task.
- `kfold_task_summary_table_full.csv`: bảng task/fold chi tiết.

## 9. Kết quả chính gần nhất

Kết quả K-Fold chính:

Thiết lập:

- Dataset: CICMalDroid, tỷ lệ malware:benign = 2:1.
- K-Fold = 5.
- Clients = 5.
- Tasks = 5.
- `task_strategy = category_dominant`.
- `dominant_ratio = 0.90`.

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.8802±0.0127 | 0.9159±0.0042 | 0.9070±0.0116 | 0.9039±0.0231 | 0.0407±0.0202 | -0.0382±0.0211 | 0.3780±0.0124 |
| FL-MalDrift | 0.8829±0.0131 | 0.9138±0.0053 | 0.9098±0.0116 | 0.9110±0.0225 | 0.0361±0.0155 | -0.0321±0.0178 | 0.3768±0.0137 |
| FL-CL-MalDrift | 0.9089±0.0101 | 0.9164±0.0031 | 0.9320±0.0087 | 0.9511±0.0183 | 0.0108±0.0126 | -0.0050±0.0157 | 0.3812±0.0183 |

Nhận xét chính:

- FL-CL-MalDrift có ACC cao nhất: 0.9089.
- FL-CL-MalDrift có F1 cao nhất: 0.9320.
- FL-CL-MalDrift có Recall cao nhất: 0.9511.
- FL-CL-MalDrift có Forgetting thấp nhất: 0.0108.
- FL-CL-MalDrift có BWT gần 0 nhất: -0.0050.
- Forgetting giảm từ 0.0407 của FedAvg xuống 0.0108 của FL-CL-MalDrift, giảm khoảng 73.5%.

Kết luận nên viết:

**FL-CL-MalDrift cho thấy hiệu quả tốt hơn trong cả hai mục tiêu: phát hiện malware và giảm catastrophic forgetting.**

## 10. Kết quả phụ để phân tích ảnh hưởng dữ liệu

### Single-run 1:1

Thư mục:

`fl_cl_maldrift_v2\results\compare_replay_only`

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.7848 | 0.8304 | 0.7521 | 0.7235 | 0.0000 | 0.0406 | 0.1878 |
| FL-MalDrift | 0.7845 | 0.8288 | 0.7529 | 0.7269 | 0.0000 | 0.1123 | 0.1396 |
| FL-CL-MalDrift | 0.8061 | 0.8392 | 0.7803 | 0.7590 | 0.0000 | 0.1038 | 0.1415 |

Nhận xét:

- FL-CL-MalDrift vẫn tốt hơn về ACC/F1/Recall.
- Forgetting chưa bật rõ vì chuỗi task chưa đủ gây quên hoặc dữ liệu cân bằng làm mô hình ổn định hơn.

### Single-run 2:1

Thư mục:

`fl_cl_maldrift_v2\results\compare_forgetting_strict_2to1`

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.9163 | 0.9154 | 0.9385 | 0.9642 | 0.0107 | 0.0003 | 0.2631 |
| FL-MalDrift | 0.9163 | 0.9147 | 0.9386 | 0.9652 | 0.0095 | 0.0013 | 0.2655 |
| FL-CL-MalDrift | 0.9217 | 0.9147 | 0.9430 | 0.9738 | 0.0046 | 0.0107 | 0.3372 |

Nhận xét:

- Tỷ lệ 2:1 làm Recall/F1 tăng do số mẫu malware nhiều hơn.
- FL-CL-MalDrift có Forget thấp hơn và BWT tốt hơn.

### K-Fold dominant_ratio = 0.75

Thư mục:

`fl_cl_maldrift_v2\results\kfold5_clients5_dominant`

| Method | ACC | Prec | F1 | Recall | Forget | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.9058±0.0119 | 0.9142±0.0117 | 0.9305±0.0084 | 0.9491±0.0075 | 0.0094±0.0045 | -0.0064±0.0083 | 0.3856±0.0118 |
| FL-MalDrift | 0.9054±0.0136 | 0.9159±0.0093 | 0.9299±0.0104 | 0.9464±0.0144 | 0.0092±0.0038 | -0.0064±0.0072 | 0.3842±0.0122 |
| FL-CL-MalDrift | 0.9109±0.0074 | 0.9163±0.0050 | 0.9343±0.0058 | 0.9547±0.0108 | 0.0086±0.0067 | -0.0034±0.0113 | 0.3946±0.0059 |

Nhận xét:

- dominant_ratio = 0.75 tạo drift vừa phải.
- Ba phương pháp khá gần nhau.
- FL-CL-MalDrift vẫn nhỉnh hơn nhưng chưa bật mạnh phần forgetting.

### K-Fold dominant_ratio = 0.90

Đây là kết quả chính nên ưu tiên viết trong báo cáo.

Nhận xét:

- Drift rõ hơn.
- Forgetting của FedAvg và FL-MalDrift tăng.
- FL-CL-MalDrift giữ Forget thấp hơn rõ rệt.
- Đây là bằng chứng tốt nhất cho phần chống catastrophic forgetting.

## 11. Vì sao FedAvg và FL-MalDrift nhiều lúc gần nhau?

Đây là điểm người dùng đã hỏi nhiều lần. Khi viết báo cáo, nên giải thích trung thực:

1. FL-MalDrift chủ yếu thay đổi aggregation, chưa có Replay/EWC.
2. Nếu drift score giữa client không quá tách biệt, trọng số drift-aware gần giống FedAvg.
3. `min_participation` hoặc cơ chế giữ đủ client làm aggregation không khác quá xa.
4. Nếu client có drift cao nhưng chứa dữ liệu malware mới hữu ích, việc down-weight có thể làm FL-MalDrift thích nghi chậm hơn.
5. Do đó FL-MalDrift không phải lúc nào cũng hơn FedAvg rõ rệt.
6. Lợi thế rõ hơn xuất hiện khi thêm Continual Learning trong FL-CL-MalDrift.

Câu nên dùng:

**FedAvg và FL-MalDrift có kết quả gần nhau vì trong thiết lập này drift-aware aggregation chưa tạo khác biệt đủ lớn nếu không có cơ chế giữ tri thức cũ như Replay và EWC.**

## 12. Cách nói về drift loop không hội tụ

Trong kết quả chính, `EscRate` và `RecRate` thường bằng 0. Vì vậy không nên nói quá rằng đã chứng minh mạnh drift loop không hội tụ bằng bảng chính.

Cách viết hợp lý:

- Bảng chính chứng minh rõ nhất phần **catastrophic forgetting**.
- Phần **drift loop không hội tụ** nên trình bày như một rủi ro thiết kế của FL-MalDrift gốc và một cơ chế đã được bổ sung trong FL-CL-MalDrift.
- DRC + Recovery Monitor được thiết kế để xử lý tình huống client drift cao bị loại/giảm ảnh hưởng lặp lại.
- Nếu cần chứng minh drift loop bằng số liệu, nên dùng thêm chế độ `--drc_stress`, vì chế độ này hạ ngưỡng K1/K2 để buộc Escalation/Recovery dễ xảy ra.

Câu nên dùng:

**Kết quả chính cho thấy FL-CL-MalDrift giảm quên tốt hơn. Riêng vấn đề drift loop không hội tụ được xử lý ở mức cơ chế thông qua DRC và Recovery Monitor; để định lượng rõ cần dùng thêm kịch bản DRC stress.**

## 13. Những nhận xét quan trọng về dữ liệu và kết quả

1. Dataset 1:1:
   - Cân bằng malware/benign.
   - Kết quả Recall/F1 thấp hơn.
   - Forgetting chưa rõ.

2. Dataset 2:1:
   - Nhiều malware hơn.
   - Recall/F1 tăng.
   - Phù hợp hơn cho bài toán phát hiện malware vì mục tiêu quan trọng là giảm bỏ sót mã độc.

3. Task strategy category_dominant:
   - Hợp lý hơn category_strict.
   - Mỗi task vẫn có đủ các họ malware, nhưng một họ chiếm đa số.
   - Mô phỏng drift theo phân phối malware thay đổi theo thời gian.

4. dominant_ratio 0.75:
   - Drift nhẹ/vừa.
   - Kết quả ba phương pháp gần nhau.

5. dominant_ratio 0.90:
   - Drift rõ hơn.
   - Làm bật forgetting.
   - FL-CL-MalDrift nổi bật nhất.

6. Precision của FL-CL có thể thấp hơn hoặc gần FedAvg:
   - Vì FL-CL nhạy hơn với malware để tăng Recall.
   - Khi malware tăng, mô hình có xu hướng bắt malware tốt hơn nhưng có thể báo nhầm benign nhiều hơn.

## 14. Nhiệm vụ Claude cần làm tiếp

Claude cần viết báo cáo/bài báo **bằng tiếng Anh**, theo đúng **mẫu/template mà người dùng đã cung cấp riêng cho Claude**. Không cần tự tạo một cấu trúc mới nếu mẫu đã có sẵn. File handoff này chỉ cung cấp technical context, số liệu, lệnh chạy và các nhận xét quan trọng để Claude đưa vào đúng vị trí trong mẫu.

Nếu mẫu của người dùng có các phần tương ứng, nên map nội dung như sau:

| Phần trong mẫu tiếng Anh | Nội dung nên đưa vào |
|---|---|
| Title / Topic | FL-CL-MalDrift: Federated Continual Learning for Adaptive Android Malware Detection Under Concept Drift |
| Abstract | Mục tiêu, phương pháp đề xuất, dataset CICMalDroid, kết quả chính |
| Introduction | Bối cảnh Android malware, federated learning, concept drift, catastrophic forgetting |
| Related Work / Baseline | FedAvg và FL-MalDrift gốc |
| Proposed Method | DRC, Replay Buffer, EWC, Recovery Monitor |
| Dataset / Experimental Setup | CICMalDroid, 1:1 và 2:1, K-Fold, task/client split |
| Metrics | ACC, Precision, Recall, F1, Forgetting, BWT, FWT |
| Results | Bảng K-Fold dominant_ratio = 0.90 là kết quả chính |
| Discussion | Vì sao FL-CL tốt hơn, vì sao FedAvg và FL-MalDrift gần nhau, ảnh hưởng của tỷ lệ dữ liệu |
| Conclusion | FL-CL cải thiện detection và giảm forgetting; drift-loop cần DRC stress để định lượng rõ hơn |

Yêu cầu khi viết bằng tiếng Anh:

- Follow exactly the report/paper template provided by the user to Claude.
- Write in academic English, clear enough for presentation/report defense.
- Use the K-Fold dominant_ratio = 0.90 result as the main result.
- Không nói quá về drift loop nếu không có EscRate/RecRate trong bảng chính.
- Nhấn mạnh rõ phần forgetting: FL-CL-MalDrift giảm Forget khoảng 73.5% so với FedAvg.
- Giải thích vì sao 2:1 giúp Recall/F1 cao hơn.
- Giải thích vì sao Precision có thể không tăng.
- Giải thích vì sao FedAvg và FL-MalDrift gần nhau.
- Viết đủ công thức cho EWC, Forgetting, BWT/FWT nếu cần.
- Nếu cần trích dẫn bài báo gốc, không tự bịa tác giả/năm; hãy yêu cầu người dùng cung cấp citation hoặc chỉ ghi là “FL-MalDrift gốc/công trình cơ sở” nếu chưa có metadata.

## 15. English writing materials for Claude

This section provides ready-to-use English content blocks. Claude should adapt them to the user's provided template, not necessarily copy them verbatim.

### 15.1 Background

Android malware detection is a continuously evolving problem because malicious applications change their behaviors, API usage patterns, and system-call characteristics over time. A detector trained on an earlier malware distribution may perform poorly when new malware families or new variants appear. This phenomenon is commonly described as concept drift, where the relationship between input features and target labels changes across time or data partitions.

In realistic deployment, malware samples may be distributed across different organizations, devices, or clients. Directly centralizing all data can be limited by privacy, ownership, and storage constraints. Federated Learning (FL) addresses this setting by allowing multiple clients to train local models and share model updates instead of raw data. However, standard FL methods such as FedAvg assume that client updates can be averaged directly, which becomes problematic when clients observe heterogeneous, non-IID, or drifting data.

Another important challenge is catastrophic forgetting. In a continual learning scenario, the model is trained sequentially on multiple tasks. After adapting to new malware distributions, the model may lose performance on previously learned malware families. Therefore, an adaptive malware detector should satisfy two goals at the same time: learning new malware patterns and preserving knowledge of old malware patterns.

### 15.2 Related Work

The related work can be organized into four groups.

First, traditional Android malware detection methods rely on static features, dynamic features, or hybrid features extracted from applications. Static analysis may use permissions, API calls, intents, manifest information, or opcode patterns. Dynamic analysis may use runtime behavior such as system calls, binder calls, network activity, or file-system events. In this project, the CICMalDroid dataset provides feature vectors based on system-call and binder-call frequency, which are suitable for supervised malware classification.

Second, Federated Learning provides a privacy-preserving learning framework where clients train local models and a central server aggregates their updates. FedAvg is the standard baseline: each client trains locally, and the server computes a weighted average of client parameters according to the number of local samples. FedAvg is simple and effective, but it does not explicitly handle concept drift, non-IID client behavior, or forgetting across sequential tasks.

Third, FL-MalDrift is the direct baseline of this project. It introduces drift-aware mechanisms into federated malware detection. Instead of treating every client update equally, the server estimates client drift scores and uses drift-aware filtering or weighting to reduce the influence of highly drifting updates. This helps the global model become more robust to distribution changes. However, the original FL-MalDrift mainly modifies aggregation and does not fully address continual learning problems such as catastrophic forgetting.

Fourth, continual learning methods such as replay and regularization are used to reduce forgetting. Replay-based methods store a small subset of previous samples and reuse them during later training. EWC, or Elastic Weight Consolidation, protects parameters that are important for previous tasks by adding a quadratic penalty weighted by Fisher information. This project combines these ideas with FL-MalDrift to build FL-CL-MalDrift.

Important citation note for Claude:

- Do not fabricate the full citation of the original FL-MalDrift paper if the metadata is not available.
- FedAvg and EWC are well-known baselines, but if the final paper requires formal references, verify the exact BibTeX/citation metadata.

### 15.3 Methodology

The proposed method is FL-CL-MalDrift, an extension of FL-MalDrift with continual learning and drift recovery mechanisms. The goal is to improve Android malware detection under concept drift while reducing catastrophic forgetting.

The method compares three training strategies:

1. FedAvg: the standard federated learning baseline. It aggregates local client models by weighted averaging based on the number of samples.
2. FL-MalDrift: a drift-aware FL baseline. It computes drift scores and adjusts aggregation through filtering or weighting.
3. FL-CL-MalDrift: the proposed method. It extends FL-MalDrift with DRC, Replay Buffer, EWC, and Recovery Monitor.

The proposed FL-CL-MalDrift includes four main components.

DRC, or Drift Resolution Controller, controls the client state under drift. It follows the sequence:

```text
Stable -> Replay -> EWC -> Escalation -> Recovery
```

When drift is mild, the client remains stable. When drift persists, the client can activate replay and EWC to stabilize learning. If drift continues for too long, the client can enter escalation, where its influence may be reduced or withheld. If the client later becomes stable again, the Recovery Monitor allows it to rejoin the federation.

Replay Buffer stores a small subset of old samples using reservoir sampling. During later tasks, these samples are reused to remind the local model of previous malware distributions.

EWC protects important parameters learned from previous tasks. The regularized objective can be described as:

```text
L_total = L_task + (lambda_EWC / 2) * sum_i F_i * (theta_i - theta_i_old)^2
```

where `L_task` is the classification loss, `F_i` is the Fisher information estimate for parameter `i`, `theta_i` is the current parameter, and `theta_i_old` is the parameter value stored after a previous task.

Recovery Monitor tracks client stability using an exponential moving average of drift scores. If a client remains stable for a required number of rounds, it can be recovered and included again in the federation.

The drift threshold `tau` is a hybrid threshold. It is initialized by command-line parameters such as `tau_init` and bounded by `tau_min`, but it is also adjusted based on observed drift statistics with smoothing. Therefore, it should be described as a configured but adaptive threshold.

### 15.4 Program Flow

The program flow is as follows:

```text
1. Load CICMalDroid CSV data
2. Preprocess numeric features
   - handle NaN/Inf values
   - remove constant columns
   - select top 200 features using mutual information
3. Convert labels to binary malware detection
   - Adware, Banking, SMS, Riskware -> Malware
   - Benign -> Benign
4. Build dataset ratio
   - 1:1 malware:benign or 2:1 malware:benign
5. If K-Fold is enabled:
   - split data into 5 stratified folds
   - each fold is an independent experiment
6. Inside each fold:
   - split training data into 5 continual tasks
   - each task contains all malware families
   - one malware family dominates each task under category_dominant
7. For each task:
   - partition task data across clients
   - run communication rounds
   - each client trains locally
   - the server aggregates client updates
8. For FL-CL-MalDrift:
   - compute drift score
   - update DRC state
   - apply Replay/EWC when needed
   - monitor escalation and recovery
9. Evaluate after each task
   - current task performance
   - previous task performance for Forgetting and BWT
10. Save outputs
   - comparison JSON/CSV
   - per-round CSV
   - per-task CSV
   - K-Fold summary CSV
   - result charts
```

The key point in the task construction is that each task is not a single malware family. Instead, every task still contains Adware, Banking, SMS, and Riskware. The difference is that one family dominates the task. This makes the drift scenario more realistic because malware distributions shift gradually while retaining diversity.

### 15.5 Experimental Setup

Dataset:

- CICMalDroid 2020.
- Five original classes: Adware, Banking, SMS, Riskware, and Benign.
- Binary conversion: Adware, Banking, SMS, and Riskware are mapped to Malware; Benign remains Benign.
- Feature selection: top 200 numeric features selected using mutual information.

Data ratios:

- 1:1 setting: 1795 malware samples and 1795 benign samples.
- 2:1 setting: 3590 malware samples and 1795 benign samples.

Main setting:

- Dataset ratio: malware:benign = 2:1.
- Number of folds: 5.
- Number of clients: 5.
- Number of continual tasks: 5.
- Rounds per task: 25.
- Local epochs: 3.
- Non-IID alpha: 0.1.
- Task strategy: category_dominant.
- Dominant ratio: 0.90.
- Replay buffer size: 500.
- EWC lambda: 0.05.
- Initial drift threshold: tau_init = 0.25.
- Minimum drift threshold: tau_min = 0.2.
- Warmup rounds: 0.

Main command:

```powershell
.\malenv\Scripts\python.exe main.py --compare --kfold 5 --dataset cicmaldroid --tasks 5 --rounds 25 --clients 5 --alpha 0.1 --local_epochs 3 --task_strategy category_dominant --dominant_ratio 0.90 --tau_init 0.25 --tau_min 0.2 --warmup_rounds 0 --ewc_lambda 0.05 --replay_buffer 500 --out results\kfold5_clients5_dominant_r090
```

Evaluation metrics:

- Accuracy: overall classification correctness.
- Precision: correctness of malware predictions.
- Recall: ability to detect malware samples.
- F1-score: harmonic mean of Precision and Recall.
- Forgetting: loss of performance on previous tasks.
- BWT: backward transfer from new tasks to old tasks.
- FWT: forward transfer from previous learning to new tasks.
- Escalation Rate and Recovery Rate: DRC behavior indicators.

### 15.6 Results

The main result is the 5-fold experiment with `category_dominant` and `dominant_ratio = 0.90`.

| Method | ACC | Precision | F1 | Recall | Forgetting | BWT | FWT |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.8802±0.0127 | 0.9159±0.0042 | 0.9070±0.0116 | 0.9039±0.0231 | 0.0407±0.0202 | -0.0382±0.0211 | 0.3780±0.0124 |
| FL-MalDrift | 0.8829±0.0131 | 0.9138±0.0053 | 0.9098±0.0116 | 0.9110±0.0225 | 0.0361±0.0155 | -0.0321±0.0178 | 0.3768±0.0137 |
| FL-CL-MalDrift | 0.9089±0.0101 | 0.9164±0.0031 | 0.9320±0.0087 | 0.9511±0.0183 | 0.0108±0.0126 | -0.0050±0.0157 | 0.3812±0.0183 |

Main observations:

- FL-CL-MalDrift achieves the highest Accuracy: 0.9089.
- FL-CL-MalDrift achieves the highest F1-score: 0.9320.
- FL-CL-MalDrift achieves the highest Recall: 0.9511.
- FL-CL-MalDrift achieves the lowest Forgetting: 0.0108.
- FL-CL-MalDrift has the BWT closest to zero: -0.0050.
- Compared with FedAvg, FL-CL-MalDrift reduces Forgetting from 0.0407 to 0.0108, which is about a 73.5% reduction.

Suggested English interpretation:

```text
The K-Fold results show that FL-CL-MalDrift consistently outperforms FedAvg and FL-MalDrift in malware detection performance. It improves Accuracy, F1-score, and Recall, indicating that the proposed continual-learning components help the model adapt to new malware distributions. More importantly, FL-CL-MalDrift substantially reduces catastrophic forgetting. Its Forgetting score decreases from 0.0407 in FedAvg to 0.0108, corresponding to an approximately 73.5% reduction. Its BWT is also much closer to zero, suggesting that learning new tasks causes less degradation on previous tasks.
```

### 15.7 Discussion Points

Why does the 2:1 malware:benign ratio improve performance?

- The model sees more malware samples during training.
- Malware Recall increases because the model becomes more sensitive to malware patterns.
- F1-score improves because Recall improves strongly.
- Precision may not always improve because a more malware-sensitive model can misclassify some benign samples as malware.

Why are FedAvg and FL-MalDrift sometimes similar?

- FL-MalDrift mainly modifies aggregation, while FedAvg uses standard averaging.
- If drift scores are not strongly separated, drift-aware weights become similar to FedAvg weights.
- If minimum client participation is enforced, many clients still contribute to aggregation.
- FL-MalDrift does not have Replay/EWC, so it cannot preserve old-task knowledge as effectively as FL-CL-MalDrift.
- Therefore, the strongest improvement appears when continual learning mechanisms are added.

How should the non-convergent drift loop be discussed?

- Do not overclaim this result from the main table because Escalation Rate and Recovery Rate are often zero.
- The main table strongly supports the catastrophic forgetting claim.
- The drift-loop issue should be discussed as a design risk handled by DRC and Recovery Monitor.
- Strong quantitative evidence for drift loop behavior requires a dedicated DRC stress experiment.

Suggested English sentence:

```text
The main experiments primarily validate the reduction of catastrophic forgetting. The non-convergent drift-loop issue is addressed at the mechanism level through DRC and the Recovery Monitor. A dedicated DRC stress setting would be required to provide stronger quantitative evidence for escalation and recovery behavior.
```

### 15.8 Program Artifacts and Outputs

Important generated artifacts:

- `BAO_CAO_FL_CL_MalDrift.docx`: Vietnamese report draft.
- `BAO_CAO_KET_QUA_FL_CL_MalDrift.pptx`: result presentation.
- `pipeline_fl_cl_maldrift.png`: program pipeline image.
- `fl_cl_maldrift_v2/results/kfold5_clients5_dominant_r090/kfold_summary.csv`: main K-Fold summary.
- `fl_cl_maldrift_v2/results/kfold5_clients5_dominant_r090/kfold_per_fold.csv`: per-fold results.
- `fl_cl_maldrift_v2/results/kfold5_clients5_dominant_r090/kfold_per_task.csv`: per-task results.

Suggested wording for the program flow figure:

```text
The pipeline begins with CICMalDroid data preprocessing, including numeric feature extraction, cleaning, binary label mapping, and feature selection. The processed data are then divided into stratified folds. Within each fold, the training split is transformed into a sequence of category-dominant continual tasks and distributed across federated clients. During training, clients perform local updates, while the server aggregates updates using FedAvg, FL-MalDrift, or FL-CL-MalDrift. For the proposed method, drift scores are monitored and the DRC activates replay, EWC, escalation, and recovery when necessary. After each task, the model is evaluated on current and previous tasks to compute detection metrics, Forgetting, BWT, and FWT.
```

## 16. Prompt có thể đưa trực tiếp cho Claude

```text
You are an academic writing assistant. Please read this CLAUDE_HANDOFF.md file and write my FL-CL-MalDrift report/paper in English, following the template I already provided to you.

Requirements:
- Write in academic English.
- Follow my provided template exactly; do not invent a new structure unless a section is missing.
- Use the K-Fold dominant_ratio = 0.90 result as the main result.
- Include dataset, proposed method, formulas/metrics, experimental setup, commands if appropriate, results, discussion, and evaluation according to the template.
- Emphasize that FL-CL-MalDrift improves ACC/F1/Recall and reduces catastrophic forgetting.
- Do not overclaim the non-convergent drift loop result. State that DRC/Recovery Monitor is the designed mechanism, and DRC stress testing is needed for clearer quantitative evidence.
- If citation metadata for the original FL-MalDrift paper is missing, mark it as needing citation instead of fabricating author/year.
```

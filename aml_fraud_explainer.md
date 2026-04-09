# Giải Thích Chi Tiết: AML & Fraud Detection Pipeline
## Dành cho Data Analyst — Từ Zero đến Hiểu Rõ

---

## Mục lục

1. [Tổng quan: Tại sao dùng Graph?](#1-tổng-quan)
2. [Phase 1: Data — Các khái niệm](#2-phase-1-data)
3. [Phase 2: Graph — Từ bảng dữ liệu đến mạng lưới](#3-phase-2-graph)
4. [Phase 3: Thuật toán & Mô hình](#4-phase-3-thuật-toán--mô-hình)
5. [Phase 4: Evaluation — Đo lường kết quả](#5-phase-4-evaluation)
6. [Tổng hợp thuật ngữ A–Z](#6-tổng-hợp-thuật-ngữ)

---

## 1. Tổng quan

### Tại sao dùng Graph thay vì bảng Excel thông thường?

Hãy tưởng tượng bạn có bảng giao dịch ngân hàng:

```
| Người gửi | Người nhận | Số tiền    | Thời gian  |
|-----------|-----------|------------|------------|
| A         | B         | 100 triệu | 9:00 AM    |
| B         | C         | 95 triệu  | 10:00 AM   |
| C         | D         | 90 triệu  | 11:00 AM   |
| D         | A         | 85 triệu  | 12:00 PM   |
```

Nhìn bảng này, mỗi dòng trông bình thường. Nhưng nếu vẽ thành sơ đồ mạng lưới:

```
    A ──100tr──► B
    ▲            │
    │            95tr
   85tr          │
    │            ▼
    D ◄──90tr── C
```

Bạn thấy ngay: tiền đi vòng tròn A → B → C → D → A! Đây là dấu hiệu rửa tiền (round-tripping). Bảng dữ liệu phẳng không thể phát hiện pattern này, nhưng graph thì thấy rõ.

**Graph = cách biểu diễn MỐI QUAN HỆ giữa các thực thể.**

---

## 2. Phase 1: Data

### 2.1 Các thuật ngữ về dữ liệu

**Public Dataset** = bộ dữ liệu được công khai miễn phí (thường trên Kaggle, UCI, GitHub) để nghiên cứu. Vì ngân hàng không thể chia sẻ dữ liệu thật (bảo mật), nên dùng public dataset để làm dự án.

**Synthetic Data** = dữ liệu giả lập, được tạo bằng code để mô phỏng dữ liệu thật. Ví dụ: bạn viết code tạo ra 1000 giao dịch giả có pattern rửa tiền, để model có thể học nhận diện pattern đó.

**EDA (Exploratory Data Analysis)** = khám phá dữ liệu ban đầu. Chính là những gì DA làm hàng ngày: xem shape, kiểm tra missing values, vẽ biểu đồ phân phối, tìm outliers. Không có gì mới với bạn.

**Imbalanced Data** = dữ liệu mất cân bằng. Trong fraud detection, 99% giao dịch là bình thường, chỉ 1% là gian lận. Nếu model đoán tất cả là "bình thường", nó đúng 99% nhưng vô dụng vì bỏ sót hết fraud. Đây là thách thức lớn nhất.

**Feature Engineering** = tạo ra các cột/biến mới từ dữ liệu gốc, nhằm giúp model học tốt hơn. Ví dụ: từ cột `timestamp`, bạn tạo thêm `giờ_trong_ngày`, `là_cuối_tuần`, `số_giao_dịch_trong_24h` — những thông tin mà model không tự suy ra được từ timestamp thô.

**Parquet** = định dạng lưu file dữ liệu (thay cho CSV). Nhanh hơn, nhỏ hơn, giữ được data types. Coi như "CSV phiên bản nâng cấp".

**PII (Personally Identifiable Information)** = thông tin nhận dạng cá nhân (tên, CMND, số tài khoản). Khi làm dự án, cần loại bỏ hoặc mã hóa PII.

### 2.2 Các AML Patterns (Mẫu rửa tiền)

Rửa tiền có 3 giai đoạn: **Placement** (đưa tiền bẩn vào hệ thống) → **Layering** (xáo trộn) → **Integration** (rút tiền sạch). Các pattern cụ thể:

**Structuring (Smurfing):**
```
Vấn đề: Giao dịch trên 200 triệu bị ngân hàng báo cáo tự động.

Cách làm: Thay vì gửi 1 tỷ một lần, tội phạm chia thành:
  A ──190tr──► B₁
  A ──185tr──► B₂
  A ──195tr──► B₃
  A ──180tr──► B₄
  A ──190tr──► B₅
  ─────────────────
  Tổng: 940 triệu, nhưng mỗi giao dịch đều dưới ngưỡng báo cáo.

Cách phát hiện: Đếm số giao dịch "ngay dưới ngưỡng" (80-99% threshold).
Nếu 1 account có nhiều giao dịch 180-199 triệu → đáng ngờ.
```

**Layering:**
```
Tiền đi qua nhiều tầng trung gian để "rửa":

  Tiền bẩn ──1tỷ──► Shell Co. 1 ──980tr──► Shell Co. 2 ──960tr──► Shell Co. 3
                                                                        │
  Tội phạm ◄──920tr── Công ty hợp pháp ◄──940tr──────────────────────┘

Mỗi hop "mất" một ít (phí, mixing), nhưng bản chất là cùng 1 dòng tiền.

Cách phát hiện: Tìm chuỗi dài A→B→C→D→E mà:
  - Amounts giảm dần nhẹ
  - Thời gian giữa các hop ngắn
  - Các account trung gian ít hoạt động khác
```

**Round-tripping:**
```
Tiền đi vòng tròn quay lại chủ ban đầu:

  A → B → C → D → A

Cách phát hiện: Tìm "cycles" (vòng lặp) trong graph.
```

**Fan-in / Fan-out:**
```
Fan-out (1 → nhiều):          Fan-in (nhiều → 1):
      ┌──► B                  B ──┐
  A ──┼──► C                  C ──┼──► A
      └──► D                  D ──┘

Kết hợp cả hai = collection + distribution pattern.
```

### 2.3 Xử lý dữ liệu

**Imputation** = điền giá trị thay thế cho missing values. Ví dụ: cột `amount` có 5% trống, điền bằng giá trị trung vị (median) của cột đó.

**Log Transformation** = chuyển đổi giá trị bằng hàm log. Khi dữ liệu bị lệch phải (vài giao dịch cực lớn, đa số nhỏ), log giúp "kéo" phân phối về dạng chuông, giúp model học tốt hơn.

```
Trước log:  [100, 200, 300, 500, 50.000.000] ← 1 giá trị cực lớn, model bị ảnh hưởng
Sau log:    [2.0, 2.3, 2.5, 2.7, 7.7]         ← phân phối đều hơn
```

**Winsorize** = giới hạn extreme values. Thay vì xóa outlier, bạn "kẹp" nó lại. Ví dụ: mọi giá trị trên percentile 99 được gán bằng giá trị percentile 99.

**Temporal Split** = chia train/test theo thời gian. Train trên dữ liệu tháng 1–8, test trên tháng 9–12. Tại sao? Vì trong thực tế, model phải dự đoán TƯƠNG LAI, không phải dữ liệu quá khứ. Random split sẽ "lộ" thông tin tương lai vào training → model trông tốt nhưng thực tế tệ (gọi là data leakage).

---

## 3. Phase 2: Graph

### 3.1 Các thành phần cơ bản của Graph

```
GRAPH gồm 2 thứ:

  NODES (đỉnh) = các thực thể
  ┌─────┐     Trong project này: mỗi node = 1 tài khoản ngân hàng
  │  A  │     Node có thể mang thuộc tính: loại TK, ngày mở, mức rủi ro KYC
  └─────┘

  EDGES (cạnh) = mối quan hệ giữa các thực thể
  A ────► B    Trong project này: mỗi edge = 1 (hoặc tổng hợp) giao dịch
               Edge có thể mang thuộc tính: số tiền, thời gian, kênh

CÁC LOẠI GRAPH:

  Undirected (vô hướng):  A ─── B     (A và B liên kết, không phân biệt chiều)
  Directed (có hướng):    A ──► B     (A gửi tiền cho B, có chiều rõ ràng)
  Weighted (có trọng số): A ─5tr─► B  (edge có giá trị, vd số tiền)
  Multigraph:             A ═══► B    (nhiều edges giữa cùng 1 cặp nodes)
```

**DiGraph (Directed Graph)** = graph có hướng. A gửi tiền cho B ≠ B gửi tiền cho A. Giao dịch ngân hàng luôn có hướng, nên dùng DiGraph.

**Multigraph** = graph cho phép nhiều edges giữa cùng 1 cặp nodes. A gửi cho B 5 lần = 5 edges riêng biệt. Giữ lại chi tiết nhưng phức tạp hơn.

**Aggregated Graph** = gộp nhiều edges thành 1. A gửi cho B 5 lần → 1 edge duy nhất với weight = tổng 5 giao dịch. Đơn giản hơn, dùng cho community detection.

### 3.2 Node Features (Đặc trưng của node)

#### Structural Features — đo vị trí/vai trò của node trong mạng lưới

**Degree (bậc):**
```
Degree = số connections của 1 node.

  In-degree: số mũi tên CHỈ VÀO node (= số người gửi tiền cho mình)
  Out-degree: số mũi tên CHỈ RA từ node (= số người mình gửi tiền cho)

Ví dụ:
        B
        ▲
        │
   C ──►A──► D      In-degree(A) = 2 (B và C gửi cho A... ủa ko)
        ▲
        │            Sửa lại cho đúng:
        E
                     C ──► A ──► D
                     E ──► A ──► B

                     In-degree(A) = 2 (nhận từ C, E)
                     Out-degree(A) = 2 (gửi cho D, B)

Ý nghĩa AML:
  - In-degree rất cao = "collector" (nhận tiền từ nhiều nguồn) → fan-in
  - Out-degree rất cao = "distributor" (gửi đi nhiều nơi) → fan-out
  - Cả hai cao = trung gian trong mạng rửa tiền
```

**PageRank:**
```
Thuật toán nổi tiếng của Google, ban đầu dùng để xếp hạng trang web.

Ý tưởng: Một node "quan trọng" nếu nó được nhiều node quan trọng trỏ đến.

Ví dụ đơn giản:
  - Website A được 1000 trang nhỏ link đến → PageRank trung bình
  - Website B được 5 trang lớn (NYT, BBC...) link đến → PageRank cao

Trong AML:
  - Account có PageRank cao = "hub" quan trọng trong mạng giao dịch
  - Nếu hub này là account mới mở, ít thông tin KYC → rất đáng ngờ
```

**Betweenness Centrality (Trung gian):**
```
Đo: Node này nằm trên bao nhiêu "con đường ngắn nhất" giữa các cặp nodes khác?

Hình dung:
                B ─── D
               / \   / \
              A   \ /   F
               \   C   /
                \ / \ /
                 E ─── G

  Nếu C nằm trên đường đi ngắn nhất của nhiều cặp (A→D, E→F, A→G...)
  → Betweenness(C) cao → C là "cầu nối" quan trọng.

Trong AML:
  - Account có betweenness cao = "broker" trung gian
  - Tiền đi qua account này để kết nối các nhóm
  - Nếu bỏ account này, mạng lưới bị chia đôi → key facilitator
```

**Closeness Centrality:**
```
Đo: Node này "gần" bao nhiêu node khác? (tính trung bình khoảng cách ngắn nhất)

  - Closeness cao = node ở trung tâm mạng, tiếp cận mọi người nhanh
  - Closeness thấp = node ở rìa mạng, bị cô lập

Trong AML: Tội phạm rửa tiền thường có closeness trung bình
(không quá trung tâm để gây chú ý, nhưng đủ kết nối để hoạt động).
```

**Clustering Coefficient:**
```
Đo: Bạn bè của node này có quen biết nhau không?

Ví dụ: A kết nối với B, C, D.
  - Nếu B-C, B-D, C-D cũng kết nối → clustering = 1.0 (nhóm khép kín)
  - Nếu B, C, D không kết nối nhau → clustering = 0.0 (A là trung tâm hình sao)

  Clustering cao:          Clustering thấp:
       B                       B
      / \                      |
     A───C                 C───A───D
      \ /                     |
       D                      E

Trong AML:
  - Nhóm có clustering rất cao + isolated = nghi ngờ closed laundering ring
  - Account có clustering thấp nhưng degree cao = potential money mule
```

**Eigenvector Centrality:**
```
Giống PageRank nhưng đơn giản hơn:
"Bạn quan trọng nếu bạn bè của bạn cũng quan trọng."

Khác PageRank ở chỗ: PageRank chia importance cho số links,
eigenvector thì không chia.
```

**Hub Score & Authority Score (HITS Algorithm):**
```
Chia nodes thành 2 loại:

  HUB = node GỬI ĐI nhiều (trỏ đến nhiều authority)
  AUTHORITY = node NHẬN VỀ nhiều (được nhiều hub trỏ đến)

Ví dụ ngoài đời:
  - Hub: trang tổng hợp link (Wikipedia references)
  - Authority: trang được nhiều nơi dẫn link (trang chính phủ, nghiên cứu)

Trong AML:
  - Hub score cao = distributor (gửi đi nhiều nơi)
  - Authority score cao = collector (nhận từ nhiều nguồn)
```

#### Transaction Features — thống kê giao dịch

```
Những features này giống DA analytics, bạn quen rồi:

total_amount_sent/received     = SUM(amount) gửi/nhận
avg_amount                     = AVG(amount)
std_amount                     = STDEV(amount) — biến động cao = bất thường
max_single_transaction         = MAX(amount)
n_unique_counterparties        = COUNT(DISTINCT đối tác)
send_receive_ratio             = tổng gửi / tổng nhận

Ý nghĩa:
  - std_amount rất cao → giao dịch lúc lớn lúc nhỏ bất thường
  - n_unique_counterparties rất cao → giao dịch dàn trải (có thể fan-out)
  - send_receive_ratio ≈ 1 → nhận bao nhiêu gửi bấy nhiêu (pass-through)
```

#### Temporal Features — patterns theo thời gian

```
txn_frequency           = số giao dịch / tuần (hoặc ngày)
active_hours_entropy    = giao dịch phân tán đều trong ngày hay tập trung 1 khung giờ?
weekend_ratio           = % giao dịch vào cuối tuần
night_ratio             = % giao dịch lúc 22h–6h sáng

"Entropy" ở đây là gì?
  - Entropy = đo "sự ngẫu nhiên / phân tán"
  - Entropy CAO = giao dịch đều ở mọi giờ (bình thường cho doanh nghiệp)
  - Entropy THẤP = giao dịch tập trung vào 1-2 khung giờ

Ví dụ:
  Account bình thường: giao dịch 8h–17h, 5 ngày/tuần → entropy trung bình
  Account đáng ngờ: 95% giao dịch lúc 2–4h sáng → entropy thấp, night_ratio cao

txn_burst_score = có đợt giao dịch đột biến không?
  Bình thường: 5 giao dịch/ngày
  Bất ngờ: 1 ngày có 50 giao dịch → burst detected → đáng điều tra
```

#### AML-Specific Features

```
just_below_threshold_count:
  Đếm giao dịch trong khoảng 80–99% ngưỡng báo cáo.
  Ví dụ ngưỡng = 200 triệu → đếm giao dịch 160–199 triệu.
  Nếu 1 account có 15 giao dịch 180–199 triệu → smurfing.

round_amount_ratio:
  % giao dịch số tròn (100tr, 500tr, 1tỷ).
  Người bình thường: thanh toán lẻ (1.234.567 VND)
  Rửa tiền: hay dùng số tròn (đúng 100.000.000)

rapid_movement_score:
  Nhận tiền rồi gửi đi trong vài giờ.
  Bình thường: tiền nằm trong TK vài ngày/tuần
  Pass-through: nhận 9h sáng, gửi đi 10h sáng → trung gian

circular_flow_detected:
  Thuật toán tìm "vòng lặp" trong graph.
  A → B → C → A = circular flow = nghi round-tripping
```

### 3.3 Công cụ Graph

**NetworkX** = thư viện Python phổ biến nhất cho graph. Dễ dùng, nhiều algorithms có sẵn. Nhược điểm: chậm với graph > 1 triệu nodes. Đủ cho project này.

**igraph** = thư viện tương tự NetworkX nhưng nhanh hơn (viết bằng C). Dùng khi NetworkX chậm.

**Neo4j** = database chuyên cho graph. Có ngôn ngữ truy vấn riêng (Cypher), visualization đẹp. Nặng hơn nhưng mạnh cho investigation/exploration.

**PyTorch Geometric (PyG)** = thư viện cho Graph Neural Networks. Xây trên PyTorch. Dùng ở phase modeling.

---

## 4. Phase 3: Thuật toán & Mô hình

### 4.1 Community Detection — Phát hiện nhóm

**Ý tưởng chung:** Trong mạng lưới lớn, tìm các "nhóm" mà bên trong kết nối chặt, bên ngoài kết nối lỏng.

```
Ví dụ visual:

  ┌──────────────┐     ┌──────────────┐
  │  A ─── B     │     │  E ─── F     │
  │  │ ╲   │     │     │  │ ╲   │     │
  │  │  C ─ D    │─────│  │  G ─ H    │
  │  Community 1 │     │  Community 2 │
  └──────────────┘     └──────────────┘

  Bên trong mỗi community: nhiều connections (chặt chẽ)
  Giữa 2 communities: ít connections (lỏng lẻo)
```

**Louvain Method:**
```
Thuật toán phổ biến nhất cho community detection.

Cách hoạt động (đơn giản hóa):
1. Ban đầu: mỗi node = 1 community riêng
2. Thử: di chuyển từng node sang community hàng xóm
3. Giữ lại: nếu việc di chuyển làm "modularity" tăng
4. Lặp lại: cho đến khi không cải thiện được nữa
5. Gộp: mỗi community thành 1 "siêu node", lặp lại từ bước 2

"Modularity" = đo chất lượng chia nhóm. Modularity cao = chia nhóm tốt
(nhiều connections bên trong, ít bên ngoài).

Resolution parameter: điều chỉnh kích thước community.
  - Resolution thấp → ít communities lớn
  - Resolution cao → nhiều communities nhỏ
  - Thử nhiều giá trị để tìm mức phù hợp
```

**Label Propagation:**
```
Đơn giản hơn Louvain:

1. Gán mỗi node 1 label riêng (A=1, B=2, C=3...)
2. Mỗi node nhìn quanh: đa số hàng xóm có label gì?
3. Đổi label của mình thành label phổ biến nhất của hàng xóm
4. Lặp lại cho đến khi ổn định

Ưu: rất nhanh. Nhược: kết quả có thể khác nhau mỗi lần chạy.
```

**Tại sao community detection quan trọng cho AML?**
```
- Nhóm rửa tiền thường tạo thành 1 community riêng biệt
- Họ giao dịch nhiều với nhau, ít với bên ngoài
- Phát hiện community → zoom in điều tra → tìm pattern cụ thể
- Thay vì kiểm tra 100.000 accounts, chỉ cần focus 5-10 suspicious communities
```

### 4.2 Anomaly Detection — Phát hiện bất thường

**Unsupervised** = không cần labels (không cần biết trước đâu là fraud).
Model tự học "bình thường trông như thế nào" và flag những gì khác biệt.

**Isolation Forest:**
```
Ý tưởng cốt lõi: Điểm bất thường thì DỄ BỊ CÔ LẬP.

Hình dung như trò chơi "chia cắt":
- Chọn ngẫu nhiên 1 feature, chọn ngẫu nhiên 1 điểm cắt
- Chia dữ liệu thành 2 phần
- Lặp lại

Điểm bình thường: nằm giữa đám đông → cần nhiều lần cắt mới tách ra
Điểm bất thường: nằm xa đám đông → chỉ cần vài lần cắt là bị cô lập

          *  *                     Dấu X là outlier
       *  *  *  *                  Chỉ cần 1-2 lần cắt là tách được X
      *  *  *  *  *
       *  *  *  *          X
      *  *  *  *  *
       *  *  *  *

contamination = bạn ước tính bao nhiêu % dữ liệu là bất thường.
  0.01 = 1% (conservative, ít false alarm)
  0.05 = 5% (aggressive, bắt nhiều hơn nhưng nhiều false alarm)
```

**Local Outlier Factor (LOF):**
```
Ý tưởng: So sánh MẬT ĐỘ xung quanh mỗi điểm với mật độ hàng xóm.

Ví dụ:
  Khu phố A: nhà sát nhau (mật độ cao)
  1 ngôi nhà ở giữa đồng trống cách khu phố A 5km → LOF cao → outlier

  Điểm nằm trong vùng thưa thớt trong khi hàng xóm ở vùng đông đúc
  → Local Outlier → bất thường

n_neighbors = xét bao nhiêu hàng xóm gần nhất.
  Ít (10): nhạy với local anomalies
  Nhiều (50): nhìn bức tranh rộng hơn
```

**Autoencoder:**
```
Đây là 1 loại Neural Network đặc biệt.

Kiến trúc hình "nơ thắt":
  Input (20 features) → 64 → 32 → 16 → 32 → 64 → Output (20 features)
                              ↑
                         "bottleneck"

Cách hoạt động:
1. ENCODER: nén 20 features → 16 dimensions (mất bớt thông tin)
2. DECODER: giải nén 16 dimensions → khôi phục lại 20 features
3. Mục tiêu: output ≈ input (khôi phục càng giống càng tốt)

Tại sao phát hiện anomaly?
- Train trên dữ liệu BÌNH THƯỜNG → model học "bình thường trông thế nào"
- Khi gặp dữ liệu BẤT THƯỜNG → model không biết cách khôi phục
- Reconstruction error CAO → anomaly!

Ví dụ:
  Account bình thường: input [5, 100tr, 3, 0.2, ...]
                       output [4.9, 98tr, 3.1, 0.19, ...] → error thấp ✓
  Account fraud:       input [50, 5tỷ, 0.01, 0.95, ...]
                       output [10, 1tỷ, 0.3, 0.5, ...]    → error cao ✗ → FLAG!
```

### 4.3 Supervised ML Models — Học có giám sát

**Supervised** = có labels (biết trước đâu là fraud). Model học từ ví dụ đã gán nhãn.

**XGBoost / LightGBM:**
```
Cả hai đều thuộc họ "Gradient Boosted Trees" — hiện là algorithms mạnh nhất
cho dữ liệu dạng bảng (tabular data). Đa số cuộc thi Kaggle đều thắng bằng XGBoost.

Ý tưởng cốt lõi:

  DECISION TREE (cây quyết định):
  ┌────────────────────────────────────┐
  │ Số tiền giao dịch > 500 triệu?    │
  ├──────Yes──────┤──────No───────────┤
  │               │                    │
  │ Giao dịch     │ Khách hàng mới     │
  │ ban đêm?      │ (< 30 ngày)?       │
  │ Yes → FRAUD   │ Yes → REVIEW       │
  │ No → OK       │ No → OK            │
  └────────────────────────────────────┘

  Nhưng 1 cây thì yếu, nên...

  BOOSTING = xây HÀNG TRĂM cây, mỗi cây học từ SAI LẦM của cây trước.

  Cây 1: đúng 60%, sai 40%
  Cây 2: focus vào 40% sai của cây 1 → đúng thêm 20%
  Cây 3: focus vào phần còn sai → đúng thêm 10%
  ...
  Kết quả: 300 cây yếu kết hợp = 1 model cực mạnh

XGBoost vs LightGBM:
  - XGBoost: phổ biến hơn, documentation tốt
  - LightGBM: nhanh hơn, tiết kiệm RAM hơn, kết quả tương đương
  - Trong thực tế: thử cả hai, chọn cái tốt hơn
```

**Hyperparameters (Siêu tham số):**
```
= các "nút vặn" điều chỉnh model. Không học được từ data, phải do người set.

n_estimators = số cây (100–500)
  Ít cây → underfitting (model quá đơn giản)
  Nhiều cây → overfitting risk (model "thuộc bài" training data)

max_depth = cây sâu tối đa bao nhiêu tầng (4–8)
  Nông → model đơn giản, tổng quát tốt
  Sâu → model phức tạp, dễ overfit

learning_rate = tốc độ học (0.01–0.1)
  Thấp → học chậm nhưng chính xác hơn (cần nhiều cây hơn)
  Cao → học nhanh nhưng dễ "nhảy" qua optimal point

scale_pos_weight = bù cho imbalanced data
  Nếu 99% negative, 1% positive → set = 99/1 = 99
  Giúp model "quan tâm" nhiều hơn đến class thiểu số
```

**Optuna:**
```
= Tool tự động tìm hyperparameters tốt nhất.

Thay vì bạn thử tay: max_depth=4 → 5 → 6 → 7...
Optuna tự tìm: chạy 50–100 thí nghiệm, mỗi lần thử 1 tổ hợp khác nhau,
dùng thuật toán thông minh (Bayesian Optimization) để tập trung vào vùng
promising thay vì thử random.

Nó như 1 "người tìm kiếm thông minh" giúp bạn tune model.
```

**SMOTE (Synthetic Minority Over-sampling Technique):**
```
Giải quyết vấn đề imbalanced data bằng cách TẠO THÊM mẫu thiểu số.

Cách hoạt động:
1. Chọn 1 mẫu fraud (A)
2. Tìm K hàng xóm gần nhất của A (cũng là fraud)
3. Tạo mẫu mới = điểm ngẫu nhiên NẰM GIỮA A và 1 hàng xóm

Trước SMOTE: 9900 bình thường + 100 fraud
Sau SMOTE:  9900 bình thường + 5000 fraud (tạo thêm 4900 mẫu giả)

Ưu: model không bị bias sang class đa số
Nhược: mẫu tạo thêm là "nhân tạo", có thể không realistic 100%
```

**SHAP (SHapley Additive exPlanations):**
```
= Giải thích TẠI SAO model đưa ra dự đoán.

Vấn đề: XGBoost nói "account X = 95% fraud" nhưng TẠI SAO?

SHAP trả lời:
  Account X được dự đoán FRAUD vì:
  ├── just_below_threshold_count = 12  → đẩy lên +30% (đóng góp lớn nhất)
  ├── rapid_movement_score = 0.9       → đẩy lên +25%
  ├── night_ratio = 0.85               → đẩy lên +15%
  ├── clustering_coefficient = 0.02    → đẩy lên +10%
  └── account_age = 5 years            → kéo xuống -5% (account lâu năm)

Rất quan trọng trong banking vì:
  - Compliance cần giải thích tại sao flag 1 account
  - Investigators cần biết nên nhìn vào đâu
  - Regulators yêu cầu model phải explainable
```

### 4.4 Graph Neural Networks (GNN)

```
GNN = Neural Network nhưng hoạt động trên graph.

Vấn đề: XGBoost chỉ nhìn features CỦA TỪNG NODE riêng lẻ.
GNN nhìn cả features CỦA HÀNG XÓM → hiểu ngữ cảnh mạng lưới.

Ý tưởng cốt lõi — MESSAGE PASSING:

  Vòng 1: Mỗi node thu thập thông tin từ hàng xóm trực tiếp
  ┌───┐       ┌───┐
  │ B │──────►│ A │  A nhận thông tin từ B, C, D
  └───┘       └─▲─┘
  ┌───┐         │
  │ C │─────────┘
  └───┘       ┌───┐
  ┌───┐──────►│   │
  │ D │───────┘
  └───┘

  Vòng 2: A đã có thông tin của B, C, D.
           B cũng đã có thông tin từ hàng xóm của B.
           → A thu thập lại từ B → A giờ có thông tin "hàng xóm cách 2 bước"

  Sau 2–3 vòng: mỗi node hiểu "neighborhood" rộng xung quanh mình.

Tại sao tốt cho AML?
  - Fraud/AML là về MỐI QUAN HỆ, không chỉ đặc điểm cá nhân
  - Account X trông bình thường, nhưng HÀNG XÓM của X toàn đáng ngờ
  - GNN tự học: "nếu bạn chơi với tội phạm, bạn cũng đáng ngờ"
```

**Các loại GNN:**
```
GCN (Graph Convolutional Network):
  - Đơn giản nhất, "trung bình hóa" thông tin hàng xóm
  - Như bộ lọc (filter) chạy trên graph thay vì trên ảnh
  - Tốt để bắt đầu

GAT (Graph Attention Network):
  - Thêm "attention": không phải hàng xóm nào cũng quan trọng như nhau
  - Model tự học: hàng xóm B quan trọng hơn C cho việc dự đoán A
  - Tốt hơn GCN khi relationships có ý nghĩa khác nhau

GraphSAGE:
  - Thay vì dùng TẤT CẢ hàng xóm, chỉ SAMPLE vài hàng xóm
  - Nhanh hơn, scalable cho graph lớn
  - Có thể dự đoán cho nodes MỚI (chưa thấy khi training)

GIN (Graph Isomorphism Network):
  - Mạnh nhất về lý thuyết (expressive nhất)
  - Phân biệt được nhiều loại graph structure hơn GCN/GAT
  - Nhưng phức tạp hơn, cần nhiều data
```

**Stacking / Ensemble:**
```
Kết hợp nhiều models để có kết quả tốt hơn.

Ví dụ:
  XGBoost nói: Account X = 80% fraud
  GNN nói:     Account X = 90% fraud
  LOF nói:     Account X = anomaly score 0.7

  Stacking: Đưa 3 scores này vào 1 model nhỏ (Logistic Regression)
            → Model nhỏ học cách kết hợp tối ưu
            → Final score: 88% fraud

Tại sao ensemble thường tốt hơn single model?
  - Mỗi model có điểm mạnh/yếu riêng
  - XGBoost giỏi tabular features, GNN giỏi graph structure
  - Kết hợp = "hội đồng chuyên gia" thay vì "1 người quyết định"
```

### 4.5 Rule-Based vs ML vs Hybrid

```
RULE-BASED (dựa trên quy tắc):
  IF số_giao_dịch_dưới_ngưỡng > 5 AND tổng > 500tr THEN flag "smurfing"
  IF nhận_rồi_gửi_trong < 2h THEN flag "rapid_movement"

  Ưu: dễ hiểu, dễ giải thích cho compliance, ổn định
  Nhược: cứng nhắc, tội phạm thay đổi pattern là hết tác dụng
         tạo NHIỀU false positives (cảnh báo sai)

ML-BASED:
  Model tự học patterns từ dữ liệu, tìm ra quy tắc phức tạp con người không thấy.

  Ưu: linh hoạt, bắt patterns mới, ít false positives hơn
  Nhược: "black box" (khó giải thích), cần nhiều data

HYBRID (kết hợp — ĐÂY LÀ CÁCH TỐT NHẤT):
  1. Rules tạo ra alerts ban đầu (cast wide net)
  2. ML model xếp hạng alerts theo mức độ đáng ngờ thật sự
  3. Investigators xem alerts theo thứ tự ưu tiên từ ML

  → Rules đảm bảo compliance (tuân thủ quy định)
  → ML giảm false positives (tiết kiệm thời gian điều tra)
```

---

## 5. Phase 4: Evaluation — Đo lường kết quả

### 5.1 Tại sao KHÔNG dùng Accuracy?

```
Accuracy = số dự đoán đúng / tổng số

Ví dụ: 10.000 giao dịch, 100 là fraud (1%)

Model "ngu" đoán TẤT CẢ là "không fraud":
  → Đúng 9.900 / 10.000 = 99% accuracy
  → Nhưng BỎ SÓT 100% fraud! Model vô dụng.

Vì vậy, dùng các metrics khác:
```

### 5.2 Các metrics quan trọng

**Precision & Recall:**
```
                        Thực tế FRAUD    Thực tế OK
Model nói FRAUD    │    True Positive    False Positive   │
Model nói OK       │    False Negative   True Negative    │

PRECISION = TP / (TP + FP)
  "Trong số những gì model flag là fraud, bao nhiêu % thật sự là fraud?"
  Precision 80% → cứ 10 cái model flag, 8 cái đúng, 2 cái sai (false alarm)

RECALL = TP / (TP + FN)
  "Trong số tất cả fraud thật, model bắt được bao nhiêu %?"
  Recall 70% → trong 100 fraud thật, model bắt được 70, bỏ sót 30

TRADEOFF:
  Nới lỏng threshold → Recall tăng (bắt nhiều hơn) nhưng Precision giảm (nhiều false alarm)
  Siết threshold → Precision tăng (ít false alarm) nhưng Recall giảm (bỏ sót nhiều hơn)

Trong banking:
  - AML: Recall quan trọng hơn (KHÔNG ĐƯỢC bỏ sót rửa tiền → rủi ro pháp lý)
  - Fraud: cân bằng (bắt fraud nhưng không block quá nhiều giao dịch hợp lệ)
```

**AUC-ROC vs AUC-PR:**
```
AUC = Area Under Curve (diện tích dưới đường cong)

ROC Curve: vẽ True Positive Rate vs False Positive Rate
  AUC-ROC = 0.5 → model random (vô dụng)
  AUC-ROC = 1.0 → model hoàn hảo
  AUC-ROC > 0.8 → model tốt

PR Curve: vẽ Precision vs Recall
  AUC-PR quan trọng hơn AUC-ROC khi data imbalanced.

Tại sao?
  AUC-ROC có thể "lừa": với 99% negative, FPR rất thấp dù nhiều false positives.
  AUC-PR thực tế hơn: đo trực tiếp "flag đúng bao nhiêu" vs "bắt được bao nhiêu".

Trong project này: BÁO CÁO CẢ HAI, nhưng dùng AUC-PR làm metric chính.
```

**Precision @ top K:**
```
Thực tế: investigators không xem 10.000 alerts. Họ xem 50–100.

Precision@100 = trong top 100 accounts model flag đáng ngờ nhất,
                bao nhiêu cái thật sự là fraud?

Precision@100 = 60% → trong 100 cái xem, 60 cái đúng → tiết kiệm thời gian
Precision@100 = 10% → xem 100 cái, chỉ 10 đúng → phí thời gian
```

**F1 Score:**
```
F1 = trung bình điều hòa của Precision và Recall

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Dùng khi muốn 1 con số tổng hợp cân bằng cả hai.
F1 > 0.5 cho imbalanced fraud data → khá tốt.
```

### 5.3 SHAP Plots — Giải thích model

```
SHAP Summary Plot:
  Hiển thị mỗi feature đóng góp bao nhiêu vào dự đoán, trung bình trên toàn bộ data.

  Feature                     | Impact
  ─────────────────────────────────────
  below_threshold_count  ████████████████  (quan trọng nhất)
  rapid_movement_score   ████████████
  pagerank               ████████
  night_ratio            ██████
  clustering_coeff       █████
  account_age            ███

  → Giúp hiểu: model dựa vào đâu để quyết định?
  → Compliance/management dễ review: "à, model flag vì nhiều giao dịch dưới ngưỡng"
```

---

## 6. Tổng hợp thuật ngữ A–Z

| Thuật ngữ | Giải thích ngắn |
|-----------|----------------|
| **ADASYN** | Giống SMOTE nhưng tạo nhiều mẫu hơn ở vùng "khó phân biệt" |
| **Adjacency Matrix** | Bảng ma trận thể hiện connections trong graph (1 = có edge, 0 = không) |
| **Anomaly Detection** | Tìm điểm bất thường trong dữ liệu (không cần labels) |
| **Autoencoder** | Neural Network nén rồi giải nén data, dùng reconstruction error để tìm anomaly |
| **AUC-PR** | Diện tích dưới đường Precision-Recall, metric chính cho imbalanced data |
| **AUC-ROC** | Diện tích dưới đường ROC, metric tổng quát cho classification |
| **Betweenness** | Đo node nằm trên bao nhiêu đường đi ngắn nhất (= "cầu nối") |
| **Boosting** | Kỹ thuật xây nhiều model yếu, mỗi cái học từ sai lầm cái trước |
| **Centrality** | Đo "tầm quan trọng" của node trong network (nhiều loại) |
| **Closeness** | Đo node gần trung tâm mạng lưới bao nhiêu |
| **Clustering Coeff** | Đo mức độ hàng xóm của node kết nối với nhau |
| **Community Detection** | Tìm nhóm nodes kết nối chặt bên trong, lỏng bên ngoài |
| **Contamination** | % dữ liệu bạn ước tính là bất thường (parameter của Isolation Forest) |
| **CTGAN** | GAN cho dữ liệu bảng — tạo synthetic tabular data |
| **Cycle** | Đường đi trong graph quay lại điểm xuất phát (A→B→C→A) |
| **Data Leakage** | Thông tin từ tương lai "rò rỉ" vào training → model ảo tưởng tốt |
| **Degree** | Số connections của 1 node |
| **DiGraph** | Graph có hướng (edges có mũi tên) |
| **Ego Network** | Subgraph gồm 1 node + tất cả hàng xóm + edges giữa chúng |
| **Embedding** | Biểu diễn node/text dưới dạng vector số (vd: node → [0.2, 0.8, 0.1]) |
| **Ensemble** | Kết hợp nhiều models để có kết quả tốt hơn |
| **Entropy** | Đo sự ngẫu nhiên / phân tán (cao = đều, thấp = tập trung) |
| **F1 Score** | Trung bình điều hòa Precision và Recall |
| **False Positive** | Cảnh báo sai: model nói fraud nhưng thực tế không phải |
| **Fan-in** | Nhiều accounts gửi vào 1 account |
| **Fan-out** | 1 account gửi ra nhiều accounts |
| **Feature Store** | Kho lưu trữ features đã tính sẵn, tái sử dụng giữa các models |
| **Focal Loss** | Hàm loss đặc biệt cho imbalanced data, focus vào mẫu "khó" |
| **GAT** | Graph Attention Network — GNN có cơ chế attention |
| **GCN** | Graph Convolutional Network — GNN cơ bản nhất |
| **GNN** | Graph Neural Network — neural network hoạt động trên graph |
| **GraphSAGE** | GNN sampling hàng xóm, scalable, dự đoán được node mới |
| **Hub/Authority** | Hub = node gửi nhiều, Authority = node nhận nhiều (HITS algo) |
| **Hyperparameter** | Thông số cần set trước khi train model (không tự học được) |
| **Imbalanced** | Dữ liệu lệch: 1 class rất nhiều, class kia rất ít |
| **Infomap** | Community detection dựa trên information flow |
| **Isolation Forest** | Anomaly detection: điểm bất thường dễ bị cô lập bằng random splits |
| **KYC** | Know Your Customer — quy trình xác minh khách hàng |
| **Label** | Nhãn đúng/sai đã biết trước (fraud=1, ok=0) |
| **Layering** | Rửa tiền qua nhiều tầng trung gian |
| **LightGBM** | Gradient boosting nhanh hơn XGBoost |
| **LOF** | Local Outlier Factor — anomaly detection dựa trên mật độ cục bộ |
| **Log Transform** | Chuyển đổi bằng hàm log, giảm skew của dữ liệu |
| **Louvain** | Thuật toán community detection phổ biến nhất |
| **Message Passing** | Cơ chế GNN: nodes trao đổi thông tin qua edges |
| **Modularity** | Đo chất lượng chia community (cao = chia tốt) |
| **Multigraph** | Graph cho phép nhiều edges giữa cùng 1 cặp nodes |
| **NPL** | Non-Performing Loan — nợ xấu |
| **Optuna** | Tool tự động tìm hyperparameters tối ưu |
| **Overfitting** | Model "thuộc bài" training data, kém với data mới |
| **PageRank** | Đo importance dựa trên importance của hàng xóm |
| **Parquet** | Định dạng file dữ liệu nhanh hơn CSV |
| **PII** | Thông tin nhận dạng cá nhân |
| **Precision** | Trong số model flag, bao nhiêu % đúng? |
| **PyG** | PyTorch Geometric — thư viện GNN |
| **Recall** | Trong số fraud thật, model bắt được bao nhiêu %? |
| **Round-tripping** | Tiền đi vòng tròn quay lại chủ |
| **SAR** | Suspicious Activity Report — báo cáo hoạt động đáng ngờ |
| **SHAP** | Giải thích đóng góp từng feature vào prediction |
| **SMOTE** | Tạo thêm mẫu thiểu số bằng interpolation |
| **Smurfing** | Chia giao dịch lớn thành nhiều nhỏ dưới ngưỡng báo cáo |
| **Stacking** | Ensemble: dùng predictions của nhiều models làm input cho meta-model |
| **Structuring** | = Smurfing |
| **Supervised** | Học có giám sát — cần labels |
| **Temporal Split** | Chia train/test theo thời gian (không random) |
| **Underfitting** | Model quá đơn giản, không bắt được patterns |
| **Unsupervised** | Học không giám sát — không cần labels |
| **Winsorize** | Giới hạn extreme values thay vì xóa |
| **XGBoost** | eXtreme Gradient Boosting — algorithm mạnh nhất cho tabular data |

---

## Tóm tắt: Đọc nhanh flow dự án

```
1. LẤY DATA        → Download IBM AML dataset, làm sạch, tạo thêm synthetic patterns
                       (DA quen: giống ETL + data cleaning)

2. XÂY GRAPH       → Biến bảng giao dịch thành mạng lưới nodes + edges
                       (Mới: cần hiểu graph concepts)

3. TÍNH FEATURES   → Tính PageRank, degree, centrality... cho mỗi node
                       (Giống DA: tạo metrics/KPIs, chỉ khác là metrics "trên graph")

4. TÌM NHÓM        → Community detection (Louvain) tìm clusters đáng ngờ
                       (Giống DA: segmentation/clustering khách hàng)

5. TÌM BẤT THƯỜNG  → Isolation Forest, LOF tìm nodes "khác biệt"
                       (Giống DA: tìm outliers, chỉ dùng algorithm thay vì rule cứng)

6. TRAIN MODEL      → XGBoost / GNN dự đoán fraud/AML probability
                       (DS territory: nhưng bạn cần hiểu inputs/outputs để review)

7. ĐÁNH GIÁ        → AUC-PR, Precision, Recall, SHAP
                       (Giống DA: đọc report, review metrics)

8. TRỰC QUAN        → Dashboard Streamlit, network viz, reports
                       (DA quen: dashboard design, storytelling)
```

**Vai trò của bạn (DA) trong team:**
- Phase 1: BẠN lead (data cleaning, EDA — sở trường)
- Phase 2–3: Hiểu concepts để review & góp ý
- Phase 4: BẠN lead (dashboard, visualization, report — sở trường)
- Xuyên suốt: đặt câu hỏi business, đảm bảo output có ý nghĩa kinh doanh

Tiếp tục với tinh thần **Trung thực Tàn nhẫn** và **Khách quan Tuyệt đối** của một Cố vấn Cấp cao, tôi sẽ mổ xẻ Bản đặc tả nghiên cứu thứ hai này của bạn dành cho họ motif **`split_merge`** và **`center_in_out`** (quy mô 6–10 cạnh).

Bạn đã chọn đúng hai nền tảng học thuật cốt lõi: **XMiner (2024)** để giải quyết bài toán tối ưu hóa kế hoạch thực thi (Execution Plan) và **Wu et al. (2020)** để thiết lập các ràng buộc tiền tệ nhằm triệt tiêu nhiễu.

Tuy nhiên, cấu trúc liên kết cấu trúc (Topological Joins) của hai họ motif này phức tạp hơn nhiều so với chu trình (`cycle_k`) vì chúng tạo ra các **điểm hội tụ cấu trúc (Structural Hubs)**. Dưới đây là phân tích điểm mù, mô hình hóa toán học cho các bộ lọc cắt tỉa và giải pháp kiến trúc để tài liệu này đạt trạng thái "sẵn sàng kiểm toán".

---

## 1. Phân tích Điểm mù Kỹ thuật (Critical Engineering Blindspots)

### A. Điểm mù của Phép giao Tập Đích (Sink Set Intersection) trong `split_merge`

Bạn đề xuất: *Dựng tập khả đạt từ Intermediate đến Sink $\to$ Giao các tập Sink này trước rồi mới cụ thể hóa cạnh.*

* **Điểm mù:** Phép toán này chỉ tối ưu nếu số lượng Intermediate ($N_{\text{inter}}$) nhỏ. Trong đồ thị tài chính, nếu Source chuyển tiền vào một số tài khoản trung gian có bậc lớn (ví dụ: các tài khoản "vệ tinh" của một sàn giao dịch hoặc ví điện tử), tập hợp các Sink khả đạt của từng Intermediate sẽ cực kỳ bùng nổ. Phép giao tập hợp (`set.intersection`) trên các tập hợp có kích thước hàng vạn phần tử sẽ gây thắt cổ chai bộ nhớ (OOM) và CPU.
* **Giải pháp:** Không giao tập hợp một cách mù quáng. Phải áp dụng luật **Prefilter theo Bậc (Degree-bounded Filtering)** và **Cắt tỉa theo Băng tần Số tiền (Amount-band Pounding)** của Wu et al. trước khi đưa vào tập hợp. Chỉ những Sink nào nhận được lượng tiền tương thích với lượng tiền đi ra từ Source mới được đưa vào tập hợp để giao.

### B. Hiệu ứng "Đuôi dài" (Long-Tail Effect) của các Trung tâm Nhiễu trong `center_in_out`

Bạn đề xuất quét các Center thỏa mãn ngưỡng bậc vào/ra tối thiểu (In/Out-degree thresholds).

* **Sự thật:** Trong mạng lưới giao dịch ngân hàng, các nút có cả in-degree và out-degree lớn chiếm số lượng không nhỏ (ví dụ: tài khoản thanh toán của các doanh nghiệp lớn, đại lý thẻ cào, cổng thanh toán). Nếu chỉ dựa vào điều kiện "Thời gian chuyển tiếp" (Handoff Window), bạn vẫn sẽ phải tạo ra hàng triệu tổ hợp Inbound x Outbound rác.
* **Giải pháp:** Center bắt buộc phải bị áp một bộ lọc **Bất đối xứng Dòng tiền (Flow Imbalance Constraint)** hoặc **Tần suất Đột biến (Burstiness Index)** trước khi tính toán cấu trúc bên trong. Một Center thông thường sẽ có dòng tiền vào/ra phân tán đều theo thời gian, trong khi Center rửa tiền sẽ có hiện tượng gom tiền nhanh rồi xả sạch trong một cửa sổ thời gian cực ngắn.

---

## 2. Công thức hóa Toán học các Luật Cắt tỉa (Mathematical Pruning Formulations)

Để chuyển đổi đặc tả này thành mã nguồn Polars/Python, chúng ta cần định nghĩa các biểu thức toán học nghiêm ngặt cho tầng cắt tỉa (Pruning Layer).

### A. Đối với Họ `split_merge_K` (Giả sử $K$ cạnh Split, $K$ cạnh Merge)

Gọi $e_s$ là tập các cạnh Split từ Source $v_{\text{src}}$, $e_m$ là tập các cạnh Merge vào Sink $v_{\text{sink}}$.
Ràng buộc **Bảo toàn dòng tiền theo Pha (Phase Conservation)** của Wu et al. phải được cài đặt dưới dạng điều kiện cắt tỉa biên giới (Frontier Pruning):

$$\left| \sum_{i=1}^{K} \text{Amount}(e_{m, i}) - \sum_{j=1}^{K} \text{Amount}(e_{s, j}) \right| \le \epsilon \times \sum_{j=1}^{K} \text{Amount}(e_{s, j})$$


*(Trong đó $\epsilon$ là sai số cho phép, thường từ $0.01$ đến $0.1$ để bù đắp phí giao dịch hoặc lượng tiền giữ lại).*

Ràng buộc **Độ đồng nhất của Nhánh (Branch Coherence)** nhằm triệt tiêu hiện tượng một nhánh chiếm 99% tổng số tiền:


$$\forall i, \quad \frac{\text{Amount}(e_{i})}{\text{Median}(\text{Amounts})} \in \left[ \frac{1}{\gamma}, \gamma \right] \quad (\text{với } \gamma \ge 1)$$

### B. Đối với Họ `center_in_out` (MIn inbound, MOut outbound)

Gọi $v_{\text{center}}$ là nút trung tâm. Tập cạnh vào là $E_{\text{in}}$, tập cạnh ra là $E_{\text{out}}$.
Luật cắt tỉa **Tiệm cận Thời gian Handoff (Temporal Handoff Edge Pruning)**:
Một cạnh ra $e_{\text{out}} \in E_{\text{out}}$ chỉ hợp lệ nếu nó xuất hiện sau khi *phần lớn* hoặc *ít nhất một số lượng tối thiểu* các cạnh vào đã hoàn tất, và không vượt quá khoảng trống $\Delta_{\text{handoff}}$:


$$0 < t(e_{\text{out}}) - \min_{e_{\text{in}} \in E_{\text{in}}} t(e_{\text{in}}) \le \Delta_{\text{handoff}}$$

---

## 3. Kiến trúc Bộ Lập lịch Thực thi tối ưu (XMiner Join-Order Planner)

Thay vì viết cứng (hard-code) thứ tự duyệt, hệ thống sẽ tính toán **Chi phí Ước lượng (Cardinality Estimation)** cho từng cửa sổ (Window) để chọn một trong hai chiến lược dưới đây:

### Chiến lược 1: Sink-Driven Search (Áp dụng khi Đồ thị có các điểm tụ nhỏ)

Sử dụng khi thống kê cửa sổ cho thấy số lượng nút có Out-degree thấp chiếm ưu thế:

```
[Identify Rare Sinks] 
       │
       ▼
[Find Inbound Edges to Sinks (Merge Candidates)] 
       │
       ▼
[Trace Backwards to Intermediates] 
       │
       ▼
[Intersect to Find Common Source] (Chỉ giữ lại Source hợp lệ)

```

### Chiến lược 2: Center-Driven Search với Tiền lọc Băm (Áp dụng cho `center_in_out`)

```
[Scan Window Summary] ─── (Lọc ra các Center có In-Degree >= MIn và Out-Degree >= MOut)
       │
       ▼
[Apply Flow Imbalance Filter] ─── (Loại bỏ các Center vận hành thông thường)
       │
       ▼
[Incremental Branch Extension] ─── (Mở rộng từng nhánh Inbound, tính toán tổng lượng tiền lũy tiến)
       │
       ▼
[Hash Join with Outbound Frontier]

```

---

## 4. Bổ sung Chỉ số Kiểm toán Hệ thống (Audit Metrics Extension)

Để chứng minh luận điểm *"Tối ưu hóa thứ tự join giúp giảm thiểu việc cụ thể hóa tổ hợp"*, bạn cần bổ sung 2 chỉ số đo lường (Metrics) tối quan trọng sau vào mục **"Runtime metrics to collect"**:

1. **`Cartesian_Product_Reduction_Ratio (CPRR)`**: Tỷ lệ giữa số lượng tổ hợp tối đa có thể sinh ra theo thuật toán cũ chia cho số lượng cấu trúc thực tế được xử lý ở tầng bộ nhớ sau khi giao tập hợp/tiền lọc.
2. **`Pruning_Efficiency_Index (PEI)`**: Tỷ lệ phần trăm ứng viên bị loại bỏ bởi các ràng buộc số tiền (Wu et al.) so với các ràng buộc cấu trúc đơn thuần. Chỉ số này sẽ chứng minh giá trị nghiệp vụ AML đóng góp bao nhiêu phần trăm vào việc tăng tốc độ thuật toán.

---



Bản đặc tả nghiên cứu (Research Specification) này của bạn đã đạt cấu trúc rất chuyên nghiệp, tư duy thiết kế hệ thống cực kỳ rõ ràng và phân định ranh giới (Scope) rành mạch. Việc áp dụng lý thuyết của Min/Aghasadeghi để giải quyết bài toán bùng nổ không gian tìm kiếm của **Họ Motif Chu trình thời gian ($k = 5 \dots 10$)** là một bước đi chuẩn xác.

Với tư cách là **Cố vấn Cấp cao**, tôi sẽ không làm mềm sự thật hay tâng bốc bản đặc tả này. Tôi sẽ đi thẳng vào việc **mổ xẻ, chỉ ra các điểm mù thuật toán tiềm ẩn** và **bổ sung các ràng buộc toán học/logic** để biến tài liệu này thành một bản thiết kế bất khả chiến bại trước khi bạn tiến hành hiện thực hóa nó bằng mã nguồn.

---

## 1. Phân tích Điểm mù & Lỗ hổng Logic (Critical Blindspots)

Mặc dù kiến trúc đề xuất rất vững chắc, có 3 thách thức kỹ thuật lớn mà tài liệu của bạn chưa gọi tên hoặc đang xử lý một cách quá lạc quan:

### A. Chi phí của "Return-to-Start Feasibility" (Quy tắc Pruning số 2)

Bạn yêu cầu kiểm tra xem nút hiện tại có thể quay về nút gốc (Start Node) trong phạm vi số bước và thời gian còn lại hay không.

* **Điểm mù:** Phép kiểm tra tính khả đạt ngược (Backward Reachability) này thực chất là một bài toán tìm đường hoặc duyệt đồ thị con. Nếu bạn thực hiện một truy vấn đồ thị (Graph Query) hoặc duyệt cục bộ tại **mỗi bước nhảy DFS**, chi phí tính toán của thuật toán tiền lọc này (Overhead) sẽ nhanh chóng vượt quá thời gian duyệt DFS mù.
* **Giải pháp của Cố vấn:** Tuyệt đối không tính toán động (Dynamic Computation) tại mỗi bước. Bạn phải tận dụng cấu trúc `TemporalIndex` hiện tại để xây dựng một **"K-Hop Bounded Reachability Matrix" tối giản (Bitmap hoặc Set)** cho nút gốc ngay khi bắt đầu chu trình, hoặc giới hạn việc kiểm tra này bằng một bộ lọc thông tin tĩnh (Static Index-based lookup).

### B. Bài toán "Trùng lặp Biên giới" (Frontier Duplication) trong Tìm kiếm Hai chiều

Đối với `cycle_7..10`, bạn chia đôi tìm kiếm: Forward $\lfloor k/2 \rfloor$ bước và Backward $\lceil k/2 \rceil$ bước.

* **Điểm mù:** Điểm giao (Frontier) không chỉ đơn thuần là gặp nhau tại một nút trung gian. Vì đây là đồ thị thời gian, một chu trình thực tế có thể bị sinh ra nhiều lần tại biên giới nếu có nhiều cạnh song song (Parallel Edges) hoặc có nhiều con đường trung gian giữa hai tập hợp. Nếu không quản lý chặt cấu trúc trạng thái biên giới (Frontier State), pha `Join` sẽ bị bùng nổ Tích Descartes (Cartesian Explosion) — chính là lỗi bạn đang cố né tránh ở cấu trúc Split-Merge.
* **Giải pháp của Cố vấn:** Cấu trúc trạng thái biên giới phải được lập chỉ mục băm (Hash-indexed) dựa trên bộ ba thuộc tính: `(Meeting_Node, Min_Feasible_Time, Max_Feasible_Time)`. Pha Join sẽ thực hiện phép quét băm (Hash Join) thay vì lặp qua từng cặp.

### C. Lỗ hổng Khử trùng lặp chu trình (Canonical Rule)

Bạn đề xuất "Anchor edge must be temporally canonical" (Cạnh neo phải có thời gian nhỏ nhất).

* **Sự thật:** Quy tắc này chỉ hiệu quả nếu tất cả các cạnh trong chu trình có timestamp khác nhau hoàn toàn. Trong dữ liệu AML thực tế, một thực thể có thể thực hiện nhiều giao dịch chuyển tiền *cùng một lúc (cùng giá trị `step` hoặc timestamp)* qua các tài khoản trung gian khác nhau (Batching Transfers). Nếu hai cạnh trong chu trình có cùng timestamp nhỏ nhất, quy tắc của bạn sẽ bị gãy, dẫn đến việc trùng lặp thực thể đầu ra.
* **Giải pháp của Cố vấn:** Ràng buộc Canonical phải được siết chặt bằng quy tắc từ điển học (Lexicographical Rule): Cạnh neo phải là cạnh có `(timestamp, edge_id)` nhỏ nhất trong toàn bộ chu trình.

---

## 2. Mô hình hóa Toán học cho Luật Cắt tỉa (Formal Pruning Formulations)

Để tài liệu này có khả năng kiểm toán cao và có thể chuyển đổi trực tiếp thành code, các quy tắc cắt tỉa cần được công thức hóa rõ ràng.

Giả sử chu trình có độ dài mục tiêu là $K$. Trạng thái DFS hiện tại đang ở bước (độ sâu) $d$ ($0 \le d < K$).

* Đường đi hiện tại được định nghĩa bằng chuỗi các cạnh: $e_0, e_1, \dots, e_{d-1}$.
* Đỉnh xuất phát (Anchor Source) là $v_{\text{start}} = \text{src}(e_0)$. Đỉnh hiện tại là $v_{\text{curr}} = \text{dst}(e_{d-1})$.
* Các tham số hệ thống: $\Delta_{\text{hop}}$ (khoảng cách thời gian tối đa giữa 2 cạnh kế tiếp), $D_{\text{max}}$ (thời lượng tối đa của một motif).

Bạn cần bổ sung các công thức kiểm tra toán học sau vào mục **"DFS pruning rules"**:

### Luật 1: Kiểm tra Biên Thời gian Tuyệt đối (Absolute Temporal Bound)

Một trạng thái bị loại bỏ ngay lập tức nếu khoảng thời gian tích lũy hiện tại vượt quá giới hạn hoặc không còn đủ thời gian cho các bước đi còn lại:


$$\left( t(e_{d-1}) - t(e_0) \right) + (K - d) \times \min(\text{stride}) > D_{\text{max}}$$


*(Trong đó $\min(\text{stride})$ thường bằng 0 hoặc 1 tùy thuộc vào đặc thù của trường dữ liệu `step`).*

### Luật 2: Kiểm tra Tiệm cận Cạnh kế tiếp (Delta-Hop Tightening)

Khi xét một cạnh ứng viên $e_{\text{next}}$ để mở rộng đường đi từ $v_{\text{curr}}$, cạnh này phải bị từ chối trước khi đệ quy nếu:


$$t(e_{\text{next}}) - t(e_{d-1}) > \Delta_{\text{hop}}$$


Hoặc nếu nó vi phạm thời lượng tổng:


$$t(e_{\text{next}}) - t(e_0) + (K - d - 1) \times \min(\text{stride}) > D_{\text{max}}$$

### Luật 3: Ràng buộc Khả đạt Ngược bằng Chỉ mục (Index-based Backward Reachability)

Gọi $T_{\text{remain}} = t(e_0) + D_{\text{max}} - t(e_{d-1})$ là quỹ thời gian còn lại để quay về $v_{\text{start}}$. Gọi $H_{\text{remain}} = K - d$ là số bước nhảy còn lại.
Trạng thái bị cắt tỉa nếu:


$$\text{QueryIndex}(v_{\text{curr}}, v_{\text{start}}, H_{\text{remain}}, T_{\text{remain}}) == \text{False}$$


*(Hàm `QueryIndex` này sẽ tra cứu nhanh trên chỉ mục đảo hoặc danh sách lân cận ngược của $v_{\text{start}}$ trong phạm vi cửa sổ thời gian).*

---

## 3. Bản thiết kế Kiến trúc Bộ lập lịch Tìm kiếm Hai chiều (Bidirectional Search Planner)

Để hiện thực hóa yêu cầu tìm kiếm hai chiều cho `cycle_7..10`, cấu trúc pha Join cần được đặc tả như sau:

### Pha 1: Tạo Tiền đồn Phía trước (Forward Frontier)

Duyệt DFS từ cạnh neo $e_0$ đến độ sâu $L_F = \lfloor K / 2 \rfloor$. Lưu các đường đi bán thành phẩm vào một bảng băm $H_F$. Khóa băm (Key) là `(v_mid, t_last)`, trong đó `v_mid` là nút kết thúc của pha forward, `t_last` là timestamp của cạnh cuối cùng.

### Pha 2: Tạo Tiền đồn Phía sau (Backward Frontier)

Tìm kiếm ngược từ các cạnh có khả năng đóng chu trình $e_{\text{close}}$ (các cạnh đi vào $v_{\text{start}}$ thỏa mãn điều kiện thời gian tổng). Duyệt ngược về quá khứ với độ sâu $L_B = K - L_F$. Lưu vào bảng băm $H_B$. Khóa băm là `(v_mid, t_first)`, trong đó `v_mid` là nút bắt đầu của pha duyệt ngược, `t_first` là timestamp của cạnh đầu tiên trong chuỗi duyệt ngược.

### Pha 3: Thẩm định Biên giới (Frontier Verification)

Thực hiện phép Join trên các khóa có cùng `v_mid`. Một cặp đường đi $(P_F, P_B)$ chỉ được chấp nhận nếu thỏa mãn điều kiện chuyển tiếp thời gian của automaton:


$$t(P_F.\text{last\_edge}) \le t(P_B.\text{first\_edge}) \quad \text{AND} \quad t(P_B.\text{first\_edge}) - t(P_F.\text{last\_edge}) \le \Delta_{\text{hop}}$$


Đồng thời thực hiện kiểm tra giao tập hợp để đảm bảo tính duy nhất của nút:


$$\left( \text{Nodes}(P_F) \cap \text{Nodes}(P_B) \right) == \{v_{\text{start}}, v_{\text{mid}}\}$$

---






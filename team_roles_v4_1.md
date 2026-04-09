# Phân Công Chi Tiết v4.1: 1 DE + 4 DS
## AML Network Analysis & Fraud Detection Project (10 tuần)
### DS-1: DA-based (biết code, muốn làm AML + ML) | DS-4: Base toán tin

---

## Tổng quan phân vai

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   DE — "Kiến trúc sư hạ tầng"                                         │
│   → Pipeline, infrastructure, dashboard Pages 1, 3, 4                  │
│   → Tích hợp outputs từ tất cả DS vào sản phẩm cuối                   │
│                                                                         │
│   DS-1 — "AML Engineer"                                     │
│   → AML rule engine, AML feature extraction, anomaly detection (IF/LOF)│
│   → Risk scoring (formula + ML), dashboard Pages 2 & 5                 │
│   → Network viz, SAR report, business review                           │
│                                                                         │
│   DS-2 — "ML Engineer"                                      │
│   → Transaction features, XGBoost/LightGBM, Optuna, SHAP              │
│   → Error analysis, threshold recommendation                          │
│                                                                         │
│   DS-3  — "Deep Learning Researcher"             │
│   → GNN (GCN, GAT), ensemble strategy, report & presentation          │
│                                                                         │
│   DS-4 (toán tin) — "Graph & Math Scientist"                           │
│   → Synthetic data (probability-driven), graph construction            │
│   → Structural features, BFS/DFS, community detection                  │
│   → Autoencoder, Mahalanobis, temporal analysis, statistics            │
│                                                                         │
│   PHÂN CHIA DS-1 vs DS-4:                                              │
│   DS-4 build DATA + GRAPH + STRUCTURAL FEATURES (nền tảng toán)        │
│   DS-1 nhận đó rồi build AML LOGIC + ML MODELS + VIZ (nghiệp vụ+ML)  │
│                                                                         │
│   REVIEW CULTURE:                                                       │
│   DS-4 review toán/thống kê của DS-1                                   │
│   DS-2 review ML pipeline của DS-1 (IF, LOF, LogReg)                   │
│   DS-3 review graph construction của DS-4 (cần cho GNN)               │
│   DS-1 review business logic & UX của tất cả                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phân chia DS-1 vs DS-4 chi tiết

```
DS-4  — BUILD NỀN TẢNG:                 DS-1  — BUILD LOGIC + ML:
═══════════════════════════════════                ═══════════════════════════════════
Synthetic data generator                           AML rule engine (OOP, 6 rules)
  (Dirichlet, Poisson, Exponential)                  (StructuringRule, CircularFlowRule...)
Graph construction (3 types)                       AML feature extraction
  (adjacency matrix, multigraph, snapshots)          (threshold_proximity, rapid_movement,
Structural features                                   fan_in/out, dùng cycles từ DS-4)
  (PageRank, centrality, eigenvector)              Anomaly detection: IF + LOF
BFS (2-hop reach)                                    (tự chạy, tự tune contamination)
DFS (cycle detection)                              Risk scoring
Community detection                                  (weighted formula + train LogReg/RF)
  (Louvain, NMI, topology classifier)             Dashboard Pages 2 & 5
Autoencoder (PyTorch)                              Network viz module (pyvis)
Mahalanobis distance                               SAR report generator (PDF)
Temporal graph analysis                            Business review toàn project
Feature validation (stat tests)
Sensitivity analysis

     DS-4 outputs ════════════════► DS-1 inputs
     ├── structural_features.parquet     → dùng trong rules, scoring
     ├── community_results.parquet       → CommunityRiskRule
     ├── cycles.parquet                  → CircularFlowRule
     ├── graph object (NetworkX)         → network viz, subgraph filter
     ├── autoencoder_scores.parquet      → behavioral_risk in scoring
     └── mahalanobis_scores.parquet      → thêm vào anomaly ensemble
```

---

## Workload so sánh

```
        Khối lượng     Độ khó          Tính chất
DE      ██████████     ███████░░░      Infrastructure + integration
DS-1    ████████████   ████████░░      AML logic + ML (sklearn) + dashboard
DS-2    ██████████     ████████░░      ML modeling + optimization
DS-3    ██████████     ██████████      Deep learning + writing
DS-4    ████████████   ████████░░      Algorithms + statistics + math

→ DS-1 và DS-4 workload lớn hơn chút (vì chia 1 mảng AML lớn cho 2 người)
```

---

## DE (Data Engineer) — Hưng

### Tóm tắt timeline

```
Tuần 1–2: Data Foundation
├── Setup Git, environments, coding standards
├── Download datasets, EDA, data dictionary
├── Cleaning pipeline (1 lệnh chạy)
└── Merge synthetic data (từ DS-4) vào final dataset

Tuần 3–4: Infrastructure & Dashboard Skeleton
├── Graph storage (Neo4j hoặc file-based)
├── Feature store (versioning, merge function)
├── Pipeline automation (Makefile)
└── Dashboard skeleton + Page 1

Tuần 5–7: Dashboard Build
├── Page 3 (Fraud Detection): alerts, threshold slider, confusion matrix
├── Page 4 (Model Performance): ROC/PR, SHAP, model comparison
└── Integrate DS-3 ensemble results, performance optimization

Tuần 8–10: Polish & Delivery
├── UI polish, error handling, loading states
├── Integrate SAR generator từ DS-1
├── Docker (bonus), screenshots, final testing
└── README + documentation

Final: ✅ Pages 1/3/4, ✅ Pipeline, ✅ Feature store, ✅ Docker
```

---

## DS-1 (AML) - Châu Anh

### Tư duy chung
DS-1 Nhận structural features + community results + cycle data + autoencoder scores từ DS-4, rồi: (1) tính thêm AML-specific features, (2) chạy anomaly detection riêng (IF + LOF), (3) build rule engine, (4) build risk scoring model, (5) code dashboard + viz. Đây là role **nhiều code nhất** trong team, nhưng mỗi phần đều mid-level — không cần deep learning.

### Tuần 1–2: Domain Research & AML EDA

```
Tuần 1:
├── Ngày 1–3: Nghiên cứu AML domain
│   ├── 3 giai đoạn rửa tiền: Placement → Layering → Integration
│   ├── Tham khảo Thông tư 35/2018/TT-NHNN
│   ├── Patterns: Smurfing, Layering, Round-tripping, Fan-in/out
│   ├── SAR report template thực tế
│   └── Output: AML patterns document (chia sẻ team)
│
├── Ngày 4–5: Deep EDA (business perspective)
│   ├── Các loại AML patterns trong dataset?
│   ├── Threshold hợp lý cho banking VN?
│   ├── Gaps: dataset thiếu gì so với real?
│   └── Output: notebooks/01_eda_business.ipynb

Tuần 2:
├── Ngày 1–2: Define synthetic data specs cho DS-4
│   ├── "Smurfing: threshold 200tr, target 5–20 receivers"
│   ├── "Layering: 3–7 hops, fee 1–3%"
│   ├── "Round-tripping: cycle 3–6 nodes, 1–7 ngày"
│   ├── "Legitimate: salary monthly, bills recurring, shopping random"
│   └── DS-4 code, DS-1 review output realistic hay không
│
├── Ngày 3–5: Plan rules + scoring + dashboard
│   ├── Draft rule specs (mỗi rule: logic, threshold, severity)
│   ├── Draft risk scoring components
│   ├── Sketch dashboard wireframes (Pages 2 & 5)
│   └── Thống nhất với DS-4: features nào DS-1 cần

Deliverables:
├── ✅ AML domain document
├── ✅ Business EDA notebook
├── ✅ Synthetic data specs → DS-4
├── ✅ Rule specs draft
└── ✅ Dashboard wireframes
```

### Tuần 3–4: AML Features, Rules & Risk Scoring

```
Tuần 3:
├── Ngày 1–3: AML feature extraction
│   ├── Module: src/aml/features.py
│   │   (NHẬN structural_features + cycles từ DS-4, TÍNH THÊM AML features)
│   │
│   ├── class AMLFeatureExtractor:
│   │   ├── extract_threshold_features(txn_df, threshold=200M):
│   │   │   ├── below_threshold_count: txn trong [80%, 100%) threshold
│   │   │   ├── below_threshold_ratio: count / total_txn
│   │   │   ├── TỰ VIẾT threshold_proximity_score:
│   │   │   │   gần threshold hơn → weight cao hơn
│   │   │   │   score = sum(amount/threshold for near txns) / n_txn
│   │   │   └── round_amount_ratio: % txn chia hết cho 1M
│   │   │
│   │   ├── extract_flow_features(txn_df):
│   │   │   ├── TỰ VIẾT rapid_movement_score:
│   │   │   │   ├── Với mỗi account: join received + sent txns
│   │   │   │   ├── Tìm pairs: recv_time < send_time < recv_time + 4h
│   │   │   │   ├── Count qualifying pairs
│   │   │   │   └── Normalize by total transactions → [0, 1]
│   │   │   │
│   │   │   ├── TỰ VIẾT fan_in_score:
│   │   │   │   = n_unique_senders * total_received / max_single_received
│   │   │   ├── fan_out_score: tương tự cho sending
│   │   │   └── net_flow = total_received - total_sent
│   │   │       (gần 0 = pass-through, âm = distributor, dương = collector)
│   │   │
│   │   ├── integrate_cycle_data(cycles_df):  ← NHẬN TỪ DS-4
│   │   │   ├── Per account: circular_flow_count, circular_flow_amount
│   │   │   └── Tạo features từ raw cycle data DS-4 cung cấp
│   │   │
│   │   └── extract_all_aml(txn_df, structural_features, cycles) → DataFrame:
│   │       ├── Merge threshold + flow + cycle features
│   │       └── Save: data/features/aml_features.parquet
│   │
│   └── Output: notebooks/03_aml_features.ipynb
│
├── Ngày 4–5: AML rule engine (OOP)
│   ├── Module: src/aml/rules.py
│   │
│   ├── class Rule (base):
│   │   ├── name, severity, description
│   │   ├── evaluate(account_features) → Alert | None
│   │   └── explain(alert) → str (human-readable)
│   │
│   ├── class StructuringRule(Rule):
│   │   ├── Logic: below_threshold_count > min_count
│   │   │         AND total > threshold * 2 AND window < 72h
│   │   ├── Severity: HIGH
│   │   └── Explain: "Account had {n} txns at 80–100% threshold..."
│   │
│   ├── class RapidMovementRule(Rule):
│   │   ├── Logic: rapid_movement_score > 0.5 AND volume > min
│   │   └── Severity: MEDIUM
│   │
│   ├── class CircularFlowRule(Rule):
│   │   ├── Logic: circular_flow_count > 0 (data từ DS-4 cycles)
│   │   └── Severity: HIGH
│   │
│   ├── class FanPatternRule(Rule):
│   │   ├── Logic: fan_in/out_score > 90th percentile
│   │   └── Severity: MEDIUM
│   │
│   ├── class NewAccountHighVolumeRule(Rule):
│   │   ├── Logic: age < 30d AND volume > 1B
│   │   └── Severity: HIGH
│   │
│   ├── class CommunityRiskRule(Rule):
│   │   ├── Logic: community_suspiciousness > threshold (data từ DS-4)
│   │   └── Severity: MEDIUM
│   │
│   ├── class AMLRuleEngine:
│   │   ├── __init__: load rules from configs/aml_rules.yaml
│   │   ├── run_all_rules(features_df) → alerts DataFrame
│   │   ├── aggregate_alerts() → per account summary
│   │   └── generate_alert_report() → stats
│   │
│   └── Config: configs/aml_rules.yaml

Tuần 4:
├── Ngày 1–2: Anomaly detection (DS-1 TỰ CHẠY IF + LOF)
│   ├── Module: src/models/anomaly_ml.py
│   │
│   ├── class AMLAnomalyDetector:
│   │   ├── run_isolation_forest(contamination=0.02):
│   │   │   ├── IsolationForest fit_predict trên all features
│   │   │   ├── decision_function → scores
│   │   │   └── Normalize to [0, 1]
│   │   │
│   │   ├── run_lof(n_neighbors=20):
│   │   │   ├── LocalOutlierFactor fit_predict
│   │   │   ├── negative_outlier_factor_ → scores
│   │   │   └── Normalize to [0, 1]
│   │   │
│   │   ├── tune_contamination(labels, values=[0.01, 0.02, 0.05, 0.1]):
│   │   │   ├── Chạy IF với nhiều contamination
│   │   │   ├── Tính precision, recall cho mỗi
│   │   │   └── Chọn best
│   │   │
│   │   ├── compare_with_ds4(ds4_ae_scores, ds4_mahal_scores):
│   │   │   ├── NHẬN autoencoder + Mahalanobis scores từ DS-4
│   │   │   ├── So sánh overlap: DS-1 methods vs DS-4 methods
│   │   │   ├── Rank correlation: Spearman giữa tất cả methods
│   │   │   └── Venn diagram: ai flag gì
│   │   │
│   │   └── ensemble_all_anomaly(weights=None):
│   │       ├── Combine: IF (DS-1) + LOF (DS-1) + AE (DS-4) + Mahal (DS-4)
│   │       ├── If weights=None: optimize via grid search
│   │       └── Final ensemble anomaly score → [0, 1]
│   │
│   └── Save: anomaly_scores_combined.parquet
│       → Dùng trong risk scoring
│       → Gửi DS-2 (extra features)
│       → Gửi DS-3 (ensemble)
│
├── Ngày 3–4: Risk scoring framework
│   ├── Module: src/aml/risk_scoring.py
│   │
│   ├── class RiskScorer:
│   │   ├── Approach 1 — Weighted formula:
│   │   │   ├── base_risk: KYC level (low=10, med=30, high=50)
│   │   │   ├── behavioral_risk: ensemble anomaly score * 100
│   │   │   ├── network_risk: f(community_suspiciousness, centrality)
│   │   │   │   (community data từ DS-4, centrality từ DS-4)
│   │   │   ├── pattern_risk: f(rules triggered, severity)
│   │   │   │   severity_weights = {LOW:1, MED:3, HIGH:7, CRIT:10}
│   │   │   │   pattern_risk = sum(weights) / max_possible * 100
│   │   │   └── final_risk = w1*base + w2*behavioral + w3*network + w4*pattern
│   │   │
│   │   ├── Approach 2 — Train ML model:
│   │   │   ├── Features: [base, behavioral, network, pattern, n_rules, ...]
│   │   │   ├── Train LogisticRegression → probability = risk score
│   │   │   ├── HOẶC RandomForestClassifier
│   │   │   └── Compare formula vs ML → pick better
│   │   │
│   │   ├── assign_tier(score):
│   │   │   Low (0–30) / Medium (31–60) / High (61–85) / Critical (86–100)
│   │   │
│   │   ├── explain_risk(account_id) → dict:
│   │   │   ├── Breakdown per component
│   │   │   ├── Top rules triggered
│   │   │   └── Human-readable narrative
│   │   │
│   │   └── evaluate_scoring(labels):
│   │       ├── AUC-ROC of risk score vs labels
│   │       ├── Precision@K (50, 100, 200)
│   │       └── "X% fraud in top Y% risk scores"
│   │
│   └── Output: risk_scores.parquet → DE, DS-3
│
├── Ngày 5: Evaluate rules vs labels
│   ├── Per rule: precision, recall
│   ├── Combined rules: overall performance
│   ├── Adjust thresholds
│   └── Output: notebooks/04_rule_evaluation.ipynb
│
└── Deliverables tuần 4:
    ├── ✅ AML features module (threshold, flow, cycles)
    ├── ✅ AML rule engine (6 rules, OOP, YAML config)
    ├── ✅ Anomaly detection IF + LOF (tự chạy, tự tune)
    ├── ✅ Anomaly ensemble (DS-1 IF/LOF + DS-4 AE/Mahal)
    ├── ✅ Risk scoring (formula + LogReg/RF, compared)
    └── ✅ Rule evaluation notebook
```

### Tuần 5–7: Dashboard, Viz & SAR

```
Tuần 5–6: Code dashboard pages
├── Page 2 — AML Investigation (DS-1 code toàn bộ):
│   ├── Search bar → account profile card
│   │   ├── Risk score + tier (color coded)
│   │   ├── Community ID + suspiciousness
│   │   ├── Key features (top 5 bất thường)
│   │   └── Rules triggered (severity badges)
│   │
│   ├── Transaction history table (sortable, filterable)
│   │
│   ├── Ego network viz (embed pyvis):
│   │   ├── Depth selector: 1-hop, 2-hop
│   │   └── Click node → jump to that account
│   │
│   ├── Risk breakdown chart:
│   │   ├── Bar chart: base/behavioral/network/pattern
│   │   └── "Tại sao account này bị flag?"
│   │
│   └── Button: "Generate SAR Report" → PDF
│
├── Page 5 — Network Analytics (DS-1 code toàn bộ):
│   ├── Community overview:
│   │   ├── Size distribution, top 10 suspicious (table)
│   │   └── Click community → show viz
│   │
│   ├── Centrality explorer:
│   │   ├── Distribution charts (dropdown chọn metric)
│   │   └── Highlight outlier nodes
│   │
│   ├── Full network overview (sampled, color by community)
│   │
│   └── Anomaly score distribution (normal vs anomalous)

Tuần 7:
├── Network visualization module
│   ├── Module: src/utils/visualization.py
│   │
│   ├── class NetworkVisualizer:
│   │   ├── plot_ego_network(G, node_id, depth=2) → HTML
│   │   │   Node size=pagerank, color=risk tier, hover=details
│   │   │
│   │   ├── plot_community(G, community_id) → HTML
│   │   │   Internal vs external edges highlighted
│   │   │
│   │   ├── plot_suspicious_pattern(G, pattern_type, accounts) → HTML
│   │   │   Annotate amounts, timestamps, flow direction
│   │   │
│   │   └── generate_viz_portfolio(top_n=10) → list[HTML]
│   │
│   ├── Static plots: risk_distribution, alert_summary, rule_effectiveness
│   └── Output: viz/ folder
│
├── SAR Report generator
│   ├── Module: src/aml/sar_report.py
│   ├── generate_sar(account_id) → PDF:
│   │   ├── Account summary (risk, tier, profile)
│   │   ├── Suspicious activity narrative (auto-generated from rules)
│   │   ├── Network diagram (SVG)
│   │   ├── Transaction timeline
│   │   ├── Risk breakdown chart
│   │   ├── Rules triggered
│   │   └── Recommended actions
│   └── Dùng: reportlab / fpdf2
│
├── Review UX toàn dashboard (Pages 1, 3, 4 của DE)
│
└── Deliverables tuần 7:
    ├── ✅ Dashboard Page 2 (AML Investigation)
    ├── ✅ Dashboard Page 5 (Network Analytics)
    ├── ✅ Network viz module + portfolio (10+ HTML)
    ├── ✅ SAR report generator (PDF)
    └── ✅ UX review feedback
```

### Tuần 8–10: Polish & Delivery

```
Tuần 8:
├── Refine risk scoring with DS-2 SHAP insights
│   ├── SHAP: feature X quan trọng → adjust rule weight?
│   ├── Error analysis: rules bắt FN mà model miss?
│   └── Iterate rules ↔ ML feedback loop
├── Edge case testing

Tuần 9:
├── Write report sections:
│   ├── AML Domain Context
│   ├── AML Rule Engine methodology
│   ├── Risk Scoring methodology
│   ├── AML Results & Findings
│   └── Business Implications
├── Review toàn report: business narrative

Tuần 10:
├── Demo prep (3 min):
│   "Account X, risk 87 — xem tại sao..."
│   Page 2 → ego network → rules → risk breakdown → SAR report

Final deliverables:
├── ✅ AML features module
├── ✅ AML rule engine (6 rules, OOP, YAML)
├── ✅ Anomaly detection IF + LOF (tự chạy + tune)
├── ✅ Anomaly ensemble (combine DS-1 + DS-4 methods)
├── ✅ Risk scoring (formula + ML, compared, evaluated)
├── ✅ Dashboard Pages 2 & 5
├── ✅ Network viz module + portfolio
├── ✅ SAR report generator
├── ✅ UX review
├── ✅ Report sections (AML domain, rules, scoring, findings)
└── ✅ 3 notebooks (EDA business, AML features, rule evaluation)
```

### DS-1 Skill set

```
Must-have:
├── Python OOP (class, inheritance, config-driven)
├── pandas (groupby, merge, agg)
├── scikit-learn (IsolationForest, LOF, LogisticRegression, RandomForest)
├── Streamlit (full page development)
├── pyvis (network visualization)
├── plotly / matplotlib
├── PDF generation (reportlab / fpdf2)
└── YAML config

Nice-to-have:
├── NetworkX basics (dùng graph objects DS-4 tạo)
├── Gephi
└── Figma / wireframing

Không cần:
├── BFS/DFS implementation (DS-4)
├── Autoencoder / PyTorch (DS-4)
├── Mahalanobis distance (DS-4)
├── Deep learning (DS-3)
└── Optuna / advanced tuning (DS-2)
```

---

## DS-2 (Fraud) - Tùng

### Tóm tắt timeline

```
Tuần 1–2: EDA & Feature Engineering
├── Research fraud detection (Kaggle, papers)
├── Deep EDA: class distribution, temporal, amount patterns
├── Feature engineering module:
│   ├── Amount features (total, avg, std, skew, percentiles)
│   ├── Counterparty features (n_unique, HHI, new_ratio)
│   ├── Temporal features (frequency, entropy, burst, velocity)
│   ├── Channel features (n_channels, switch_count)
│   └── Behavioral change features (amount_change, frequency_change)
├── Merge: structural (DS-4) + transaction (DS-2) + AML (DS-1)
└── Output: all_features.parquet

Tuần 3–4: Data Prep & Model Training
├── Temporal split (70/15/15 by time)
├── Feature selection, leakage check
├── Imbalance: raw+class_weight vs SMOTE vs ADASYN
├── XGBoost baseline → Optuna 100 trials → tuned model
├── LightGBM → compare → pick winner
└── Output: tuned models (.pkl)

Tuần 5–7: Evaluation, SHAP & Error Analysis
├── Evaluation module: AUC-ROC, AUC-PR, P@K, calibration
├── SHAP: summary, bar, dependence, force, waterfall plots
├── Integrate DS-1 anomaly scores → retrain → improvement?
├── Support DS-3 ensemble
├── Error analysis: top 20 FP, top 20 FN
│   ├── DS-1 rules bắt được FN mà model miss?
│   └── Output: error_analysis.md
└── Deliver: predictions, SHAP, evaluation → DE

Tuần 8–10: Threshold & Delivery
├── Threshold selection: "100 alerts/day → best threshold?"
├── Help DS-3 finalize ensemble, review report
├── Demo prep (3 min): model performance + SHAP

Final: ✅ Fraud model, ✅ Evaluation suite, ✅ SHAP,
       ✅ Error analysis, ✅ Feature pipeline, ✅ Threshold rec.
```

---

## DS-3  — Deep Learning Researcher — Giữ nguyên

### Tóm tắt timeline

```
Tuần 1–2: Learn GNN + Support
├── Self-study PyTorch + PyG (Cora tutorial)
├── Read GNN fraud papers
├── Convert dataset to PyG format (phối hợp DS-4 graph)
└── Module: src/models/gnn_data.py

Tuần 3–4: GNN Development
├── GCN baseline: class FraudGCN
│   2 GCNConv + classifier, BCEWithLogitsLoss, early stopping
├── GAT: class FraudGAT (multi-head attention)
├── Compare GCN vs GAT
├── Hyperparameter tuning
├── GraphSAGE (optional)
└── Modularize: src/models/gnn_model.py

Tuần 5–7: Ensemble + Report
├── Ensemble module:
│   ├── Collect: xgb (DS-2), gnn (self), IF/LOF (DS-1),
│   │           AE/Mahal (DS-4), rules (DS-1)
│   ├── Stacking: meta-learner on VALIDATION predictions
│   ├── Weighted average: optimize weights
│   ├── Rank-based
│   ├── Ablation: remove 1 model → impact
│   └── Grand comparison table
│
├── Report (15–20 pages):
│   ├── Ch1 Introduction (DS-3)
│   ├── Ch2 Background (DS-3 + DS-4 graph theory + DS-1 AML)
│   ├── Ch3 Data (DS-4 drafts)
│   ├── Ch4 Methodology (all contribute):
│   │   4.1 Graph construction + structural features (DS-4)
│   │   4.2 Community detection (DS-4)
│   │   4.3 AML features (DS-1)
│   │   4.4 AML rules (DS-1)
│   │   4.5 Risk scoring (DS-1)
│   │   4.6 Transaction features (DS-2)
│   │   4.7 Supervised ML (DS-2)
│   │   4.8 Anomaly detection (DS-1 IF/LOF + DS-4 AE/Mahal)
│   │   4.9 GNN (DS-3)
│   │   4.10 Ensemble (DS-3)
│   ├── Ch5 Results (all contribute)
│   ├── Ch6 Discussion (DS-3)
│   └── Ch7 Conclusion (DS-3)

Tuần 8–10: Presentation & Final
├── Presentation deck (15–20 slides)
├── Rehearsal: DS-3(3min) + DS-4(3min) + DS-1(3min) + DS-2(3min) + DE(2min) + Q&A(1min)
├── Q&A preparation

Final: ✅ GNN models, ✅ Ensemble, ✅ Grand comparison,
       ✅ Report (final), ✅ Slides, ✅ Q&A prep
```

---

## DS-4 (Graph & Statistics) - Phong

### Tư duy chung
DS-4 Build graph, implement thuật toán đồ thị, structural features, community detection, autoencoder + Mahalanobis (anomaly nặng toán). DS-4 **không** chạy IF/LOF (DS-1 làm), **không** build rules/scoring (DS-1 làm), **không** build dashboard (DS-1/DE làm). DS-4 focus 100% vào data + graph + algorithms + statistics.

### Tuần 1–2: Synthetic Data & Statistical EDA

```
Tuần 1:
├── Ngày 1–2: Nghiên cứu graph-based fraud detection
│   ├── Papers: graph theory cho financial networks
│   ├── Community detection algorithms (toán: modularity optimization)
│   ├── Anomaly detection on graphs (density estimation, reconstruction)
│   └── Output: literature_notes_graph.md
│
├── Ngày 3–5: Statistical EDA
│   ├── Degree distribution: power law? (fit test)
│   ├── Amount distribution: log-normal? Pareto?
│   ├── Temporal: Poisson process?
│   ├── Graph density, diameter ước lượng
│   └── Output: notebooks/01_eda_statistical.ipynb

Tuần 2:
├── Ngày 1–4: Code synthetic AML generator
│   ├── Module: src/data/synthetic.py
│   │   (DS-1 define business specs, DS-4 code probability models)
│   │
│   ├── class SyntheticAMLGenerator:
│   │   ├── generate_smurfing(n_cases=50):
│   │   │   Amount: Dirichlet allocation + clip + redistribute
│   │   │   Timestamp: Poisson process in time window
│   │   │   Noise: Gaussian perturbation
│   │   │
│   │   ├── generate_layering(n_cases=30):
│   │   │   Chain length: Poisson(λ=4) + 3
│   │   │   Fee: Uniform(1%, 3%) per hop
│   │   │   Delay: Exponential(λ=12h)
│   │   │   Amount decay: geometric sequence
│   │   │
│   │   ├── generate_round_trip(n_cases=20):
│   │   │   Cycle length: Discrete Uniform(3, 6)
│   │   │   Amount noise: Normal(0, 0.05*amount)
│   │   │
│   │   ├── generate_fan_pattern(n_cases=30):
│   │   │   Degree: Poisson(λ=15)
│   │   │   Amount split: Dirichlet
│   │   │
│   │   ├── generate_legitimate(n_accounts=1000):
│   │   │   Salary: Normal, monthly
│   │   │   Shopping: Log-normal
│   │   │   Activity: mixture of Gaussians
│   │   │
│   │   └── generate_full_dataset():
│   │       Merge, assign IDs, validate
│   │
│   └── Statistical validation:
│       ├── QQ-plots: synthetic vs real
│       ├── KS test: p-values
│       └── Unit tests: constraints satisfied
│
├── Ngày 5: Validate + merge (phối hợp DE, review DS-1)

Deliverables:
├── ✅ Synthetic generator (probability-driven, KS-validated)
├── ✅ Statistical EDA notebook
├── ✅ Unit tests
└── ✅ Literature notes
```

### Tuần 3–4: Graph Construction, Features & Community

```
Tuần 3:
├── Ngày 1–3: Graph construction
│   ├── Module: src/graph/builder.py
│   │
│   ├── class TransactionGraphBuilder:
│   │   ├── build_multigraph() → nx.MultiDiGraph
│   │   ├── build_aggregated_graph() → nx.DiGraph
│   │   │   Adjacency matrix: A[i][j] = weight
│   │   ├── build_temporal_snapshots(freq='W') → dict
│   │   │   Track: |V(t)|, |E(t)|, density(t)
│   │   ├── get_graph_stats(G):
│   │   │   |V|, |E|, density, components, degree stats
│   │   │   Degree distribution fit: power law test
│   │   └── filter_subgraph(G, node_ids, depth=2):
│   │       BFS-based extraction
│   │
│   └── Output: notebooks/02_graph_exploration.ipynb
│       Degree distribution (log-log), component sizes, temporal evolution
│
├── Ngày 4–5: Structural feature extraction
│   ├── Module: src/graph/features.py
│   │
│   ├── class GraphFeatureExtractor:
│   │   ├── extract_centrality_features():
│   │   │   degree, in_degree_ratio, weighted_degree
│   │   │   pagerank: π = αAπ + (1-α)e
│   │   │   betweenness, closeness, eigenvector
│   │   │   hub/authority (HITS)
│   │   │
│   │   ├── extract_neighborhood_features():
│   │   │   clustering_coefficient
│   │   │   TỰ VIẾT: avg_neighbor_degree, neighbor_degree_std
│   │   │   TỰ VIẾT: _bfs_reach(start, max_depth=2):
│   │   │       BFS with deque, return |visited| - 1
│   │   │   TỰ VIẾT: local_density (ego subgraph density)
│   │   │   n_triangles
│   │   │
│   │   └── extract_all() → DataFrame (tqdm)
│   │
│   └── Save: structural_features.parquet → DS-1, DS-2

Tuần 4:
├── Ngày 1–3: Community detection
│   ├── Module: src/graph/community.py
│   │
│   ├── class CommunityDetector:
│   │   ├── detect_louvain(resolution): modularity optimization
│   │   ├── detect_label_propagation()
│   │   │
│   │   ├── compare_methods(resolutions=[0.5, 1.0, 1.5, 2.0]):
│   │   │   NMI: 2*I(X;Y) / (H(X)+H(Y)) between methods
│   │   │   Select best (max modularity + stable NMI)
│   │   │
│   │   ├── profile_communities(features_df):
│   │   │   Per community: |V|, |E|, density, volume
│   │   │   Inter/intra ratio, avg centrality
│   │   │   TỰ VIẾT suspiciousness_score:
│   │   │       = w1*density_z + w2*volume_z + w3*(1-inter_ratio) + w4*avg_anomaly
│   │   │
│   │   └── TỰ VIẾT topology_classifier(subgraph):
│   │       star_score = max_degree / avg_degree
│   │       chain_score = longest_path / |V|
│   │       clique_score = density
│   │       Classify: argmax → star(smurfing), chain(layering), clique(ring)
│   │
│   └── Output: community_results.parquet → DS-1, DE
│
├── Ngày 4–5: DFS cycle detection + deliver
│   ├── TỰ VIẾT detect_circular_flows(G, max_length=6):
│   │   DFS-based:
│   │   def find_cycles(G, max_len):
│   │       for start in G.nodes():
│   │           stack = [(start, [start])]
│   │           while stack:
│   │               node, path = stack.pop()
│   │               for nbr in G.successors(node):
│   │                   if nbr == start and len(path) >= 3:
│   │                       cycles.append(path)
│   │                   elif nbr not in path and len(path) < max_len:
│   │                       stack.append((nbr, path+[nbr]))
│   │
│   │   Per cycle: nodes, total_amount, total_time
│   │   Save: cycles.parquet → DS-1 (cho CircularFlowRule)
│   │
│   └── Deliver:
│       → DS-1: structural_features, community_results, cycles
│       → DS-2: structural_features, community_id per account
│       → DS-3: graph object (for PyG conversion)
│       → DE: community stats (for dashboard)
│
└── Deliverables tuần 4:
    ├── ✅ Graph builder (3 types, stats, power law test)
    ├── ✅ Structural features (centrality + BFS + neighborhood)
    ├── ✅ Community detection (multi-method, NMI, topology classifier)
    ├── ✅ DFS cycle detection
    └── ✅ All outputs delivered
```

### Tuần 5–7: Anomaly Detection (Toán nặng) & Analysis

```
Tuần 5:
├── Ngày 1–4: Autoencoder + Mahalanobis (DS-4 EXCLUSIVE)
│   ├── Module: src/models/anomaly_math.py
│   │
│   ├── class MathAnomalyDetector:
│   │   ├── run_autoencoder(encoding_dim=16, epochs=50):
│   │   │   ├── PyTorch implementation:
│   │   │   │   class Autoencoder(nn.Module):
│   │   │   │       encoder = Sequential(Linear(n,64), ReLU, Linear(64,32),
│   │   │   │                           ReLU, Linear(32,16))
│   │   │   │       decoder = Sequential(Linear(16,32), ReLU, Linear(32,64),
│   │   │   │                           ReLU, Linear(64,n))
│   │   │   │   Loss: MSE, Optimizer: Adam
│   │   │   │   Early stopping on validation loss
│   │   │   ├── Score: per-sample reconstruction MSE
│   │   │   └── Normalize to [0, 1]
│   │   │
│   │   ├── run_mahalanobis_distance():
│   │   │   ├── D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
│   │   │   ├── Handle: singular covariance (regularization: Σ + εI)
│   │   │   ├── Chi-square: D² ~ χ²(p) under normality
│   │   │   ├── P-value: p < 0.01 → anomaly
│   │   │   └── Normalize D² to [0, 1]
│   │   │
│   │   └── compare_ae_mahal():
│   │       ├── Correlation between scores
│   │       ├── Overlap in top anomalies
│   │       └── Each vs labels: precision, recall
│   │
│   └── Save: ae_scores.parquet, mahal_scores.parquet
│       → DS-1 (combine with IF/LOF for ensemble)
│       → DS-3 (for grand ensemble)
│
├── Ngày 5: Feature validation
│   ├── Normality tests: Shapiro-Wilk, Anderson-Darling
│   ├── Multicollinearity: VIF scores
│   ├── Feature correlations: heatmap
│   ├── Recommendations to DS-2: "drop these features"
│   └── Output: notebooks/05_feature_validation.ipynb

Tuần 6:
├── Ngày 1–3: Temporal graph analysis
│   ├── Graph metrics over time (from snapshots):
│   │   density(t), avg_degree(t), n_components(t)
│   ├── TỰ VIẾT change point detection:
│   │   CUSUM or simple z-score based detection
│   │   "Week 23 had unusual structural change"
│   ├── Community stability: NMI between consecutive periods
│   ├── New nodes/edges growth rate
│   └── Output: notebooks/06_temporal_analysis.ipynb
│
├── Ngày 4–5: Deliver all outputs
│   → DS-1: ae_scores, mahal_scores (for anomaly ensemble)
│   → DS-2: feature validation, drop recommendations
│   → DS-3: all scores (for ensemble), graph for PyG validation
│   → DE: temporal metrics (for dashboard if desired)

Tuần 7:
├── Ngày 1–2: Validate DS-3 PyG conversion
│   ├── edge_index matches original graph?
│   ├── Node features aligned?
│   └── Graph properties preserved?
│
├── Ngày 3–5: Write report sections
│   ├── Graph Theory Background (Ch2)
│   ├── Graph Construction methodology (Ch4.1)
│   ├── Structural Features (Ch4.2)
│   ├── Community Detection (Ch4.3)
│   ├── Anomaly Detection: AE + Mahalanobis (Ch4.8 phần math)
│   ├── Graph Analysis Results (Ch5.3)
│   └── Data Description (Ch3)
│
└── Deliverables tuần 7:
    ├── ✅ Autoencoder anomaly scores
    ├── ✅ Mahalanobis anomaly scores
    ├── ✅ Feature validation report
    ├── ✅ Temporal graph analysis
    ├── ✅ PyG data validation
    └── ✅ Report sections drafted
```

### Tuần 8–10: Robustness & Delivery

```
Tuần 8:
├── Sensitivity analysis:
│   ├── Rules: results change with ±20% threshold?
│   ├── Community: stable across resolutions?
│   ├── Anomaly: stable across contamination?
│   └── Report: robustness analysis section
├── Validate DS-1 risk scoring mathematically
│   "Weights sum to 1? Scores bounded? Edge cases?"

Tuần 9:
├── Review report: math accuracy, notation consistency
├── Verify statistical claims
├── Help DS-3 methodology writing (graph notation)

Tuần 10:
├── Demo prep (3 min):
│   ├── "Graph construction: N nodes, M edges, power law distribution"
│   ├── "Community detection: K communities, top 3 suspicious"
│   ├── "Anomaly comparison: Venn diagram"
│   └── "Cycle detection: found X circular flows"

Final deliverables:
├── ✅ Synthetic data generator (probability, KS-validated, tested)
├── ✅ Graph builder (3 types, statistical properties)
├── ✅ Structural features (centrality + BFS + neighborhood)
├── ✅ Community detection (multi-method, NMI, topology)
├── ✅ DFS cycle detection
├── ✅ Autoencoder anomaly detection (PyTorch)
├── ✅ Mahalanobis distance anomaly detection
├── ✅ Temporal graph analysis
├── ✅ Feature validation
├── ✅ Sensitivity / robustness analysis
├── ✅ Report sections (graph, features, community, anomaly math)
└── ✅ 5 notebooks (stats EDA, graph, community, anomaly, temporal)
```

### DS-4 Skill set

```
Must-have:
├── Python OOP
├── NetworkX (graph construction, algorithms)
├── numpy / scipy (linear algebra, statistics)
├── PyTorch basics (for Autoencoder)
├── Graph algorithms: BFS, DFS (tự implement)
├── Statistics (hypothesis testing, distributions)
└── matplotlib (distribution plots)

Toán tin advantage:
├── Linear algebra (eigenvectors → PageRank, Mahalanobis)
├── Probability (Dirichlet, Poisson → synthetic data)
├── Optimization (modularity → community detection)
├── Information theory (entropy, NMI)
├── Multivariate statistics (covariance, Mahalanobis)
└── Algorithm design (DFS cycles, BFS reach)

Không cần:
├── Streamlit / dashboard
├── SHAP / business explainability
├── AML domain / business rules
├── XGBoost tuning / Optuna
└── Technical report writing (chỉ math sections)
```

---

## Interaction Map

```
Tuần 2:
  DE ──cleaned data──────────────────────► ALL DS
  DS-1 ──synthetic specs────────────────► DS-4 (code implementation)
  DS-4 ──synthetic data──────────────────► DE (merge)
  DS-1 ──review synthetic────────────────► DS-4 (realistic?)

Tuần 3:
  DS-4 ──NetworkX graph──────────────────► DS-3 (PyG conversion)
  DS-4 ──graph stats─────────────────────► DS-1 (business interpretation)

Tuần 4:
  DS-4 ──structural features─────────────► DS-1 (AML features, rules, scoring)
  DS-4 ──structural features─────────────► DS-2 (merge into feature matrix)
  DS-4 ──community results───────────────► DS-1 (CommunityRiskRule, risk scoring)
  DS-4 ──community_id per account────────► DS-2 (extra feature)
  DS-4 ──cycles.parquet──────────────────► DS-1 (CircularFlowRule, flow features)
  DS-4 ──community stats─────────────────► DE (dashboard)

Tuần 5:
  DS-1 ──risk scores─────────────────────► DE (dashboard Page 2)
  DS-1 ──anomaly scores (IF+LOF)─────────► DS-2 (extra features)
  DS-1 ──anomaly scores (IF+LOF)─────────► DS-3 (ensemble)
  DS-1 ──rule scores─────────────────────► DS-3 (ensemble)
  DS-2 ──XGBoost predictions─────────────► DE (dashboard Page 3)
  DS-2 ──SHAP values─────────────────────► DE (dashboard Page 4)
  DS-2 ──XGBoost + LightGBM scores───────► DS-3 (ensemble)
  DS-4 ──AE + Mahalanobis scores─────────► DS-1 (anomaly ensemble)
  DS-4 ──AE + Mahalanobis scores─────────► DS-3 (grand ensemble)

Tuần 6:
  DS-1 ──combined anomaly ensemble───────► DS-2 (retrain with extra features)
  DS-4 ──feature validation──────────────► DS-2 (drop recommendations)
  DS-2 ──feedback on features────────────► DS-4, DS-1

Tuần 7:
  DS-3 ──ensemble results────────────────► DE (dashboard Page 4)
  DS-1 ──viz HTML files──────────────────► DE (dashboard Pages 2, 5)
  DS-1 ──AML sections────────────────────► DS-3 (report)
  DS-2 ──model sections──────────────────► DS-3 (report)
  DS-4 ──graph/math sections─────────────► DS-3 (report)
  DE   ──architecture diagram────────────► DS-3 (report)

Tuần 8:
  DS-1 ──SAR generator───────────────────► DE (dashboard)
  DS-4 ──sensitivity analysis────────────► DS-3 (report)
  DS-2 ──SHAP insights───────────────────► DS-1 (refine rules/scoring)

Tuần 9–10:
  DS-3 ──report + slides─────────────────► ALL (review + present)
```

---

## Presentation (15 min + 5 min Q&A)

```
DS-3: Intro + methodology + GNN + ensemble        (3 min)
DS-4: Graph analytics + anomaly (AE/Mahal)         (3 min)
DS-1: AML findings + dashboard demo (Page 2)       (3 min)
DS-2: Model performance + SHAP                     (3 min)
DE:   Dashboard overview + live demo               (2 min)
DS-3: Conclusion + Q&A                             (1 min + Q&A)
```

---

## Risk Mitigation

| Rủi ro | Giải pháp |
|--------|----------|
| DE pipeline chậm tuần 1–2 | Deliver parquet trước, polish sau |
| DS-1 rules không bắt fraud | ML model (DS-2) là backup |
| DS-1 ↔ DS-4 handoff delay | Pair review tuần 2, 4 — DS-4 deliver features trước ngày 3 tuần 4 |
| DS-2 model overfit | Temporal split, cross-val, regularization |
| DS-3 GNN không converge | Fallback: GNN embeddings + XGBoost |
| DS-4 Autoencoder fail | DS-1 vẫn có IF + LOF, 2 methods đủ |
| DS-4 cycle detection chậm | max_length=5, sample starting nodes |
| DS-3 report không kịp | Outline tuần 4, mỗi người draft tuần 6–7 |

**Critical path:**
```
DS-4 (graph + features) ──tuần 4──► DS-1 (AML logic + anomaly)
                                     DS-2 (merge features + train)
                         ──tuần 3──► DS-3 (PyG conversion + GNN)
```
DS-4 là người giao hàng sớm nhất — nếu DS-4 chậm, 3 người còn lại bị block.

# Nasdaq Fraud Detection：提交项目面试解读

核对日期：2026-10-22。依据 `submission/code/`、实际提交 PDF、`output/metrics/`、已保存模型及提交 Parquet。没有重新训练或修改提交件；外部数据只用于检查交付完整性。

## 1. 一句话说清项目

这是一个交易级、交易批准前的欺诈识别问题。项目通过训练集 EDA 发现金额与交易前余额的强关系，比较三种分类器和四组特征，以内部 calibration AP 选择模型、以 calibration F1 选择阈值，最终提交使用 23 个输入特征的加权逻辑回归。

核心贡献是业务关系特征及对异常高分的审慎验证。最终模型没有使用图特征，没有使用深度学习，也没有直接调用 Snap ML 的完整 GFP。

## 2. 提交文件与职责

| 文件 | 职责 |
|---|---|
| `submission/code/EDA.ipynb` | 训练数据探索入口；调用 `plot/`、`GFP/` 中的函数 |
| `submission/code/plot/` | 数据质量、类别、金额/余额、时间漂移、账户行为、原系统 flag 和图统计的 EDA |
| `submission/code/GFP/graph_features.py` | 五个严格早于当前小时的交易次数、对手方数和 pair 次数 |
| `submission/code/GFP/gfp_additional_features.py` | 七个历史金额和反向交易次数特征 |
| `submission/code/experiments/eda_summary.py` | 重现主要数据质量摘要和 EDA 图 |
| `submission/code/experiments/train_ablation.py` | 主流程：特征、预处理、12 组实验、阈值、选模、全量重训及外部预测 |
| `submission/code/experiments/cv_3fold.py` | 两组特征 × 三种分类器的三折分层稳定性检查 |
| `submission/code/experiments/temporal_validation.py` | 事后设置时间边界的顺序敏感性检查 |
| `submission/code/experiments/reference_rules.py` | 同一 holdout 上的全负类、原 flag、全余额规则参考线 |
| `submission/code/experiments/report_figures.py` | 读取已保存指标、评分和模型，生成报告图 |
| `submission/fraud_detection_report.pdf` | 面向技术和业务读者的最终报告 |
| `submission/test_fraud_external.parquet` | 原测试数据 + `is_fraud_predicted` |

代码中的 `ROOT` 对提交版解析为 `submission/`。按 README 从仓库根目录执行时，应保留显式 `--train`、`--test`、`--output` 参数；无参数默认路径并不全指向仓库根目录。

## 3. 数据与探索发现

- 训练集：3,144,097 行，357 笔 fraud，3,143,740 笔合法交易。
- fraud 比例为 0.0113546%，每十万笔约 11.35 笔；全预测合法也有约 99.9886% accuracy。
- 保存的质量检查记录：无缺失、无重复整行、无常数原始列。
- 五种交易类型；本训练样本的 fraud 只在 CASH_OUT（250 笔）和 TRANSFER（107 笔）出现。这不证明其他类型未来绝不会有 fraud。
- 账户 ID 高基数；不直接 one-hot，而是保留文档定义的 C/M 角色，以及候选历史统计。
- `hours_elapsed` 是跟踪起点后的小时数，没有真实日历起点，不能可靠推导工作日或实际钟点。
- fraud 出现在小时 635–743，合法交易截至小时 718。719–743 为全正类尾段，无法检验对合法交易的误报。
- 文档把原系统标记写为 `is_fraud_flagged`，实际 Parquet 字段为 `is_flagged_fraud`；代码以实际字段为准。
- EDA 可以观察交易后余额，但批准前模型明确排除两个 `*_balance_after` 字段。

最重要的 EDA 规则：设 A 为交易金额，B 为发起人交易前余额，`B > 0 且 abs(A-B) <= 0.01`。全训练文件中有 346 个命中，全部标记 fraud，占 357 个 fraud 中的 346 个。其余 11 个是三个零金额且零起始余额的 CASH_OUT 和八个 TRANSFER。

这个规则使用的是交易前字段，本身没有把目标当输入。但它是在看过全训练集标签后发现的，所以后来随机切出的 holdout 已经受到“特征发现”的间接暴露。

## 4. 特征如何帮助简单模型

| 组合 | 输入列数 | 内容 |
|---|---:|---|
| raw | 10 | elapsed hour、type、amount、两个交易前余额、system flag、四个 C/M 角色指示 |
| raw_plus_extra | 23 | raw 加 13 个金额、余额、零值和类型交互特征 |
| raw_plus_graph | 22 | raw 加 12 个历史图/账户统计 |
| raw_plus_extra_plus_graph | 35 | 上述两组都加 |

23 是预处理前列数，交易类型 one-hot 后的模型维数会变化。

extra 包括金额 log1p；发起人与收款人零余额；金额/两个起始余额；金额减发起人余额及其绝对值；金额超过余额；全余额命中；零金额；金额和起始余额同时为零；TRANSFER/CASH_OUT 与全余额指示的交互。

比例只在分母大于零时计算，其余设 NaN；数值中位数在 fit 分区内估计，零余额另外有指示。真实零值没有被当缺失删除。差值和绝对差值把“接近全额转走”变成直接可用的模型输入，交互项把类型与这种关系联系起来。逻辑回归对输入特征是线性的，但这些人工构造的输入已经包含非线性关系。

## 5. 图特征到底做了什么

每笔交易是 initiator → recipient 的有向边。

五个基础特征：发起人此前转出笔数、不同收款人数；收款人此前收款笔数、不同发起人数；当前有向 pair 此前交易次数。

七个补充特征：发起人此前转出金额的 sum/mean/max；收款人此前转入金额的 sum/mean/max；此前反向 B→A 的次数。最后一个只表示两边关系中的反向历史，不是任意长度环路检测或完整资金流分析。

对小时 h，只使用小时 < h 的记录。同一小时的所有交易使用同一历史快照，因为没有小时内部真实顺序。基础实现用累计计数减去当前小时计数；金额按账户-小时聚合再去掉当前小时；反向交易用 `merge_asof(..., allow_exact_matches=False)`。

历史特征在随机切分前基于整个训练交易表构建，不读取标签。其可接受性取决于部署是否允许较早的未标注交易更新历史状态，不能无条件宣称没有跨分区影响。最终选择的逻辑回归没有使用 graph 列。

## 6. 训练、选模、阈值各自做什么

主实验随机种子 42，分层拆分为：

| 分区 | 行数 | fraud 数 | 用途 |
|---|---:|---:|---|
| fit 60% | 1,886,457 | 214 | 拟合预处理与分类器 |
| calibration 20% | 628,820 | 72 | 比较模型 AP，选择 F1 阈值 |
| holdout 20% | 628,820 | 71 | 评价内部表现 |

数值用 median imputation + StandardScaler，类型用众数填补 + OneHotEncoder(handle_unknown='ignore')，统一装进 Pipeline。所有可学习预处理只在模型 fit 数据上拟合。

模型为 balanced Logistic Regression、XGBoost、LightGBM。两个 boosting 模型用负/正样本比作为 `scale_pos_weight`，主实验约 8,814。没有 SMOTE，没有系统超参数搜索；每个分类器在不同特征组间维持相同配置。

AP 用连续 fraud 分数评估排序。候选模型在 calibration AP 上比较，再以 calibration F1 和较少特征数打破平局。逻辑回归与 XGBoost 在 23 列组合上都得到 calibration AP/F1=1，代码在完全平局时因固定遍历顺序保留逻辑回归；并非证明逻辑回归普遍优于 XGBoost。

每个候选阈值由 calibration precision-recall curve 最大化 fraud F1 得到。这里 calibration 是选择阈值的数据分区，不是实施了概率校准。最终 cutoff 为 0.9992964880111456；类别加权和强分离使分数靠近 1，不能把这个数解释成真实 fraud 风险 99.93%。

代码对所有候选计算并保存 holdout 指标，但选择分支只使用 calibration。calibration 被反复用于比较及选阈值，加上此前全文件 EDA，均限制其独立性。

## 7. 实验结果应如何讲

以下来自保存的 CSV 和 holdout 连续评分重新核对：

| 特征 / 分类器 | Holdout AP | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---|
| raw / Logistic Regression | 0.190729 | 0.666667 | 0.140845 | 0.232558 | 10 / 5 / 61 |
| raw / XGBoost | 0.776468 | 0.730159 | 0.647887 | 0.686567 | 46 / 17 / 25 |
| raw + extra / Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 71 / 0 / 0 |
| raw + extra / XGBoost | 1.000000 | 1.000000 | 0.985915 | 0.992908 | 70 / 0 / 1 |
| raw + graph / XGBoost | 0.760551 | 0.847826 | 0.549296 | 0.666667 | 39 / 7 / 32 |
| raw + extra + graph / Logistic Regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 71 / 0 / 0 |

额外金额/余额特征的作用很强；graph 在选中的组合上没有显示增益。LightGBM 的固定配置出现严重分数饱和/误报问题，不能据此推断 LightGBM 不适合 fraud。需要进一步检查权重、学习率、叶子约束及训练稳定性；这些是待做实验，不是已经证明的解释。

全余额参考规则在同一 holdout 命中 70/71 fraud，无误报。选中模型命中 71/71，比规则多一个；不能因此断言已独立证明额外恢复能力，因为规则发现也受全文件标签暴露。

AP=1 与阈值 recall<1 并不矛盾：raw+extra XGBoost 的分数把 fraud 排在合法交易前面，但 calibration 冻结的阈值仍可高于某个 holdout fraud 分数。

## 8. 两种补充验证

三折分层 CV 只测 raw 和 raw+extra，不测 graph。每个外折内部再划 calibration 来选阈值，然后用整个 outer training 重新拟合，再评价外折。这也说明了一个限制：重训可能改变分数尺度，旧阈值不一定保持完全相同的业务表现。

raw+extra 逻辑回归的三折均值为 AP 0.9999766573、Precision 1.0、Recall 0.9831932773、F1 0.9914893617。单次 holdout 的完美结果并未在每个外折的召回上完全重复。

时间敏感性：fit 小时 0–659；calibration 660–680；holdout 681–718，共 14,939 行、64 fraud。raw 逻辑回归 AP 0.03625，8,452 FP；加入 extra 后 AP=1、64 TP、0 FP/FN；同时去掉 elapsed hour 和 system flag 的版本也相同。

这表明该时段内强分离不需要 hour 和 flag，但边界是看完标签时间覆盖后设置的，主导特征也来自全文件 EDA，因此属于事后敏感性分析，不是独立未来测试。不能把它称为部署效果的证明。

## 9. 最终预测流程与交付核查

选好特征、算法及阈值后，用全部 3,144,097 个训练样本重拟合 Pipeline。此后才读取外部测试数据，应用相同特征定义和冻结的阈值。外部共 2,247,484 行，829 个正类预测，约 0.0369%；这只是预测率，不是可验证的真实 fraud 率。

已按每批 100,000 行比较原外部 Parquet 与提交件的全部原始字段：值和顺序均一致；原索引仍在；只有业务预测列被追加；预测全为 0/1。Parquet 中的 `__index_level_0__` 是原有 pandas 索引存储，不是额外业务特征。

外部没有可用真值，不能报告它的 AP、Precision、Recall 或 F1。最终模型只保存约 7 KB，与线性模型和小型预处理器相符；模型大小不是性能或安全性的证据。

## 10. 面试必须主动说明的局限

1. 完美成绩基于 71 个 holdout fraud，支持数小；不能许诺未来零误报。
2. 全文件 EDA 先发现全余额关系，再切分，所以 holdout 并非对特征假设完全独立。
3. 随机划分混合时间和共享账户，未独立证明未来时间或新账户泛化。
4. 时间敏感性是 post-hoc，不是预注册的未来盲测。
5. `is_flagged_fraud` 的来源与生成时刻需要业务确认，不能默认批准前一定可用。
6. 最终选中的模型不使用 graph；graph 的历史状态可用性仍应单独明确。
7. 数值分数未校准。全量重训复用冻结阈值也存在分数尺度转移风险。
8. 没有给定真实 alert budget/cost，选择 F1 是实验代理目标；提交用固定阈值，不是 top-K。当前没有保存 Precision@K/Recall@K 实验结果，不能声称做过。
9. 没有单独去除 full-balance 特征的独立消融，也没有只去 flag 的专门对照；时间实验同时去掉 hour 和 flag。
10. 历史实验使用 Python 3.9.25；当前迁移到 Python 3.12.15 并完成环境烟雾检查，没有在新环境重跑整套实验。

优先改进：确认预测时点和字段来源；冻结特征与阈值选择方案；获得真正未用于 EDA 的后续混合类别数据；分开评估 full-balance、flag、hour 和 graph 的贡献；按真实审查能力评价 Precision@K/Recall@K，并检查分数校准与漂移。

## 11. 报告中的两处数值不一致

- PDF 的 raw+extra XGBoost holdout AP 写为 0.9993；保存 CSV 及 holdout 分数重算均为 1.0。0.99926 对应的是该模型的三折平均 AP，不能混为单次 holdout AP。
- PDF 文本写选中模型 holdout ROC-AUC=0.999999997；保存 CSV 及 holdout 分数重算均为 1.0。0.9999999973 对应选中模型三折平均 ROC-AUC。

两者不改变选模结论，但面试应采用上述可复核的口径。本次没有修改已经提交的 PDF。

## 12. 可练习的项目开场

“我把它定义为交易批准前的交易级 fraud 检测。训练数据有三百多万笔，但只有 357 个 fraud，所以我用 AP 和 fraud precision/recall，而不是 accuracy。EDA 发现交易金额相对于发起人交易前余额有很强信号，我用比例、差值、全余额指示和交易类型交互来表达，并在同一拆分下比较三种分类器及图历史特征。最终选择加权逻辑回归加金额/余额特征；单次内部 holdout 检出 71 个 fraud、没有误报，三折平均 F1 约 0.9915。图特征没有给最终组合带来增益。不过主导特征是在全文件 EDA 中发现的，数据也有明显时间覆盖偏差，因此我把高分解释为当前数据内部的可分性，后续需要真正独立的未来样本验证。最后我冻结阈值、全量重训，并生成保留原始外部数据的预测交付件。”

这段话是依据当前项目事实整理的练习稿；请用自己实际参与和能够解释的表述作答。

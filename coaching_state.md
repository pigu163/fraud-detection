# Interview Coaching State

## Profile
- Company: Nasdaq
- Role: Data Scientist recruitment challenge; exact job description and seniority not provided.
- Interview timeline: 2026-10-23, Asia/Shanghai (user said tomorrow on 2026-10-22).
- Language: Chinese for explanations; interview language not yet provided.
- Current request: Understand the whole fraud project, especially submission contents, before interview practice.

## Active Coaching Strategy
- Current stage: Project comprehension and evidence verification; no mock answers scored yet.
- Approach: Explain actual submission pipeline, feature rationale, validation, metrics, and reliability limitations before drills.
- Highest-priority next step: Walk through train_ablation.py and rehearse an accurate two-minute project introduction.
- Evidence: submission/code, submitted PDF, saved experiment metrics, model metadata and pipeline, stored holdout scores.
- Detailed guide: output/interview/project_walkthrough_zh.md.

## Verified Project Facts
- Selected model: class-weighted logistic regression; raw_plus_extra (23 input columns); threshold 0.9992964880111456.
- Training: 3,144,097 rows / 357 fraud; random holdout: 628,820 rows / 71 fraud.
- Selected holdout: AP=1, ROC-AUC=1; TP=71, FP=0, FN=0, TN=628749.
- Three-fold selected mean AP=0.9999766573, mean F1=0.9914893617, mean recall=0.9831932773.
- External: 2,247,484 rows; 829 positive predictions. All original values and row order checked against submitted Parquet.
- Risks to explain: full-file feature discovery exposure, temporal coverage, small positive support, flag provenance, uncalibrated scores, refit threshold transfer, conditional graph-history availability.
- Report discrepancies: raw_plus_extra XGBoost holdout AP is 1.0 in saved scores, not PDF 0.9993; selected holdout ROC-AUC is 1.0, not PDF 0.999999997. Those near-one values correspond to CV means.

## Coaching Notes
- The user wants a concrete understanding of their submitted work ahead of tomorrow's interview.
- The user is comfortable with precision, recall, and F1 and wants to focus next on AP, ROC-AUC, alert counts, and evidence limits.
- Do not claim specific Nasdaq interview practices without sources; none researched this session.
- Do not invent project outcomes, operational savings, alert-budget experiments, or candidate answers.

## Session Log
| Date | Activity | Outcome |
|---|---|---|
| 2026-10-22 | Submission-focused project review | Created evidence-grounded Chinese walkthrough; identified two reporting discrepancies; no submission files changed. |
| 2026-10-23 | Metrics and technical interview preparation | Explained the report's ranking and operational metrics; prepared 30 challenge-specific likely questions grouped by data, features/graph, validation, and reliability. |
| 2026-10-23 | Written technical answer bank | Added evidence-based short answers to all 30 challenge questions for spoken rehearsal. |

## Fraud Challenge Interview Answer Bank

These are likely **project-specific practice questions**, not confirmed Nasdaq interview questions. Answers describe the submitted experiments; they should not be presented as future deployment results. Source of record: `submission/fraud_detection_report.pdf`, `submission/code/experiments/`, and saved metrics in `output/metrics/`.

### Project and data

1. **请用两分钟介绍这个项目：要解决什么问题、怎么做、得到了什么结果？** 这是逐笔交易的欺诈检测任务。我只用有标签的训练文件做 EDA、特征工程和验证，比较四组特征与 Logistic Regression、XGBoost、LightGBM 三种模型。金额与交易前余额的关系是最强信号；选中的加权 Logistic Regression 在分层留出集的 71 笔欺诈中检出 71 笔、误报 0 笔。由于特征规律在全训练文件上被发现，这个结果仅说明文件内部区分度很高，不能当作未来效果保证。

2. **为什么预测单位是一笔交易，而不是账户或 AML 案件？** 标签 `is_fraud` 和提交要求都是逐笔交易，最终每一行生成一个 `is_fraud_predicted`。账户 ID 可以用于构造此前交易历史，但没有账户级或洗钱案件级真值，不能把评估对象改成账户或 AML 阶段。

3. **3,144,097 笔训练交易只有 357 笔欺诈，这对建模和评估有什么影响？** 欺诈率约 0.0114%，普通准确率容易掩盖漏检。必须看 AP、欺诈类 Precision/Recall/F1，以及误报与漏报的绝对数量；验证集还必须有足够正例，且每项指标使用本分区的分母。

4. **为什么 99.9886% 的 accuracy 仍可能对应完全没用的模型？** 把所有交易都判成正常，就能得到这个准确率，却抓不到任何一笔欺诈：Recall 为 0。多数类极多，让总体正确率看起来很高。

5. **EDA 中最重要的发现是什么？346/357 的 full-balance 规律如何定义？** 令 A 为交易金额、B 为付款方交易前余额；当 B>0 且 |A−B|≤0.01 时，记为 full-balance match。全训练文件有 346 笔满足该规则的欺诈，占 357 笔欺诈中的 346 笔；这些数字来自全文件 EDA，不是独立留出集的泛化估计。

6. **其余 11 笔不符合规律的欺诈是什么情况？如何避免写死规则？** 其中 3 笔是零金额、零起始余额的 CASH_OUT，另 8 笔是 TRANSFER。我保留零值和金额关系等一般性特征，让模型学习，并用独立分区比较；不根据这 11 行的身份写特例。全文件查看标签本身仍使后续验证存在发现暴露。

7. **`hours_elapsed` 能说明一天中的几点或星期几吗？为什么？** 不能。它是经过的小时编号，缺少日历起点；同一小时内也没有可信的交易先后顺序。因此只把它当排序和验证时间轴，不推断真实钟点或星期。

### Features and graph

8. **四组 feature 分别是什么？为什么做四组消融？** Raw 10 是交易类型、金额、交易前余额、elapsed hour、系统 flag 与双方 C/M 角色；Extra 13 是金额、余额比值、差值、零值和交易类型交互；Graph 12 是更早小时的账户与交易对历史。四组是 Raw、Raw+Extra、Raw+Graph、Raw+Extra+Graph，分别检验金额余额和 graph 是否有增量价值；数字是一热编码前的列数。

9. **金额／起始余额的比值遇到余额为零时怎么处理？** 只在分母大于零时计算比值；否则返回缺失值，并保留单独的零余额指示。模型 pipeline 中的中位数填补只在拟合分区学习，再应用到验证分区，避免出现无穷大或预处理泄漏。

10. **为什么不能使用交易后的余额？** 这是审批前评分场景，交易后的余额在决策时尚不可用。把它放进模型会利用未来信息。提交模型仅使用交易前余额。

11. **12 个 graph 特征覆盖哪些历史行为？** 五个计数：付款方历史出交易次数、不同收款方数，收款方历史入交易次数、不同付款方数，以及同一 A→B 交易对次数；六个金额统计：双方相应方向的历史总额、均额、最大额；最后一个是此前 B→A 的反向交易次数。它们是局部历史统计，不是完整图模式挖掘。

12. **`degree`、`fan` 和同一交易对次数有何区别？** 在这份实现里，degree 是此前相应方向的交易次数，fan 是不同对手方数量，同一交易对次数只数指定的 A→B。一个账户可以对同一对手方转多次，所以 degree 与 fan 不一定相同。

13. **反向交易计数代表什么？为什么不能直接解释为洗钱环路？** 当前 A→B 时，统计更早小时里 B→A 的次数；它只是二节点反向互动。没有资金来源、受益人、案件标签或更长路径证据，不能据此声称发现洗钱环路。

14. **为什么只用更早小时，同一小时不互相更新？** 同一小时交易的真实先后未知。如果使用同小时其他交易，可能把预测时尚未发生的行为当历史。实现只读取 `hour < 当前小时`，把同小时视为一个批次。

15. **Graph 特征没提升选中模型的 AP，怎样解释？** 在这份固定配置和分层留出集上，Raw+Extra 的 Logistic Regression 已达到 AP 1；再加 Graph 没有可见增益。这不证明 graph 方法普遍无效，可能是余额规律过强、正例太少或图历史信号弱。当前选择简单模型更合理，未来应在真正新的时间段做更有判别力的消融。

16. **随机切分前从全训练表构建历史特征，什么条件下合理？** 只有在上线时能够持续获得此前发生的**无标签**交易来更新历史状态时，这才与在线评分相符。实现禁止未来小时和目标标签进入特征，但跨随机分区的历史可用性仍是需要明确验证的业务假设。

### Models, validation, and metrics

17. **为什么选 Logistic Regression、XGBoost 和 LightGBM？** 它们覆盖可解释线性基线及两种常见表格树模型。在相同特征分组与固定主要配置下比较，可以先判断特征贡献；本实验没有做超参数搜索，不能以一次 LightGBM 的差结果评价整个模型家族。

18. **为什么给少数类加权？对分数有什么影响？** 正例仅 357 笔，加权使训练时欺诈错误承担更高损失。它可能提升少数类识别，但也会改变输出分数的数值分布；加权后的 `predict_proba` 不能直接当作校准后的真实欺诈概率，因此阈值必须在校准分区选。

19. **主实验 60/20/20 怎样切？各分区做什么？** 两次 `StratifiedShuffleSplit`、seed 42：先留 20% holdout，再从剩下的 80% 中取四分之一作 calibration，得到整体 60% fit、20% calibration、20% holdout。对应 1,886,457/214、628,820/72、628,820/71（行数/欺诈数）；fit 训练，calibration 选模型和阈值，holdout 最后评估。

20. **模型和阈值分别怎样选？为什么不能看 holdout 再调？** 每个候选在 calibration 上找欺诈类 F1 最高的阈值；候选模型按 calibration AP 排名，同分依次看 calibration F1 与较少的特征数。选定后锁住阈值再看 holdout；若用 holdout 调参，报告的 holdout 指标会偏乐观。

21. **AP 怎样计算？为什么作主指标？** 对每笔交易的连续欺诈分数从高到低排序，沿阈值变化累加“新增加的 Recall × 当时的 Precision”，即 `sum((R_n−R_(n−1))P_n)`。它反映真正欺诈能否在充满正常交易的名单里尽早出现；代码调用 `average_precision_score`，不是梯形插值 PR 面积。

22. **AP、ROC-AUC、P、R、F1 各回答什么问题？** AP 看整体高分名单的质量；ROC-AUC 看随机欺诈是否排在随机正常交易之前。固定阈值后，P 问报警有多准，R 问抓到多少欺诈，F1 综合两者。前两个用连续分数，后三个受阈值影响。

23. **为什么 AP=1 时固定阈值的 Recall 仍可能小于 1？** AP=1 说明排序可以完美把欺诈放到正常交易前面；阈值若设得过高，排在前面的某些欺诈仍低于它。项目中的 Raw+Extra XGBoost 在主 holdout 有 AP=1，但固定阈值下只检出 70/71。

24. **71 TP、0 FP、0 FN 应怎样准确表述？** “在这一个 628,820 笔交易、71 笔欺诈的内部分层留出集上，模型发出 71 个报警，全部为欺诈，没有漏检。”随后补充：只有 71 个正例，而且特征规律在全文件标签上发现过，不能保证未来零误报或零漏检。

25. **3 折验证增加什么证据？为什么不能代替未来时间验证？** 三个分层随机外层验证折都显示金额余额特征的增益；选中模型平均 AP 约 0.99998、平均 Recall 约 0.9832、平均 F1 约 0.9915。这降低单个随机划分偶然性的担忧，但各折仍混合时间，且共享此前全文件的特征发现，不能证明未来稳定。

26. **时间验证怎样切？为什么 719 小时以后单独处理？** 在小时 0–659 拟合（3,126,823 行/215 欺诈），660–680 选择阈值（2,294/37），681–718 评估（14,939/64）。719–743 只有 41 笔欺诈而没有正常交易，可看是否检出，却不能衡量误报和正常的 Precision；边界是事后确定的。

27. **Raw 抓到 64/64 却误报 8,452，怎样向业务解释？** “它没有漏掉这 64 笔欺诈，但让调查团队处理 8,516 个报警，其中 8,452 个是正常交易，报警几乎无法使用。”同一时间区间加入 Extra 后仍检出 64/64、误报为 0；这只是事后的文件内敏感性结果，不是未来工单量承诺。

### Credibility and deployment

28. **全文件发现 full-balance 规律如何影响 holdout 信心？** 规则本身仅用审批前字段，不是直接把标签当输入；但我在划分前利用全训练文件标签发现了它，留出集已参与假设形成。因而结果可描述文件内可分性，不能称为完全独立的特征发现验证。下一步应先冻结特征，再评估之后的新时间段。

29. **`is_flagged_fraud` 的产生时间为何重要？若来自上游审批决定怎么办？** 如果 flag 在当前审批决定前不可得，使用它就不符合预设预测时间，可能形成泄漏或复制上游决策。应向数据提供方确认产生时间及逻辑，必要时剔除并单独做消融。时间实验同时移除了 hour 和 flag，尚不能孤立量化 flag 的贡献。

30. **再给一轮实验和未来数据，先做哪三个验证？上线前还需什么？** 第一，预先固定特征、时间边界与阈值，在全新且同时含两类的后续时段做前向测试。第二，分别移除 full-balance 指示和 system flag，测 AP、Recall、误报增量。第三，检查分数校准、群体/时间漂移与 graph 历史在线可用性。上线决策还需真实的审核容量、误报成本、漏检成本，以及字段在审批前是否可用。

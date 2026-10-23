<!-- ppt-master-schema: design-spec/v1 -->
# Fraud Detection - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | fraud_detection |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 20 |
| Primary Language | en-US |
| Target Audience | Data science interviewers and technical reviewers |
| Communication Intent | Explain the complete project, beginning with EDA, then feature ablation, models and results, followed by metrics and validation |
| Desired Audience Outcome | Understand why features help and distinguish random-partition stability from prospective validity |
| Core Message / Ask / Action | Amount/balance features deliver the main measured gain, while discovery exposure and temporal coverage limit generalization |
| Delivery Context | Live interview, approximately 12–15 minutes |
| Artifact Afterlife | Editable interview deck stored in ppt/ |
| Reading Mode | presentation |
| Content Strategy | Balanced default, keep source facts and the requested narrative order |
| Design Style | Evidence-led editorial, monochrome |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — delegated proactive policy |
| Custom Animations | disabled — delegated proactive policy |
| Narration Audio | disabled — delegated proactive policy |
| Created Date | 2026-10-23 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 64px horizontal, 48px vertical |
| Content Area | 64,48 to 1216,672 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: Develop the requested EDA-to-feature-to-model argument, show results, then explain metrics and validation. Use topic titles for methods and supported findings for evidence pages. Distinguish observations from generalization claims.
- **Visual style**: custom
- **Visual Style Behavior**: Flat black field, white Arial typography, large open margins, sparse horizontal rules, evidence images, editable data charts and relationship diagrams. Retain compact tables only where exact lookup is useful. Original EDA figures keep their source colors and white figure backgrounds inside black slides.
- **Theme**: Monochrome editorial evidence
- **Tone**: Precise, restrained and candid

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #000000 | All slides |
| Secondary background | #111111 | Table header or emphasized row |
| Primary | #F5F5F5 | Titles and primary values |
| Accent | #FFFFFF | Key evidence |
| Secondary accent | #D0D2D6 | Comparison emphasis |
| Body text | #F5F5F5 | Main copy |
| Secondary text | #A7ABB2 | Context and sources |
| Divider | #36383D | Rules |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | Clear grotesk, bold | Arial | | sans-serif |
| Body | Neutral sans-serif | Arial | | sans-serif |

- **Title stack**: Arial
- **Body stack**: Arial

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 26 |
| Title | 44 |
| Subtitle | 30 |
| Annotation | 20 |
| Table | 23 |
| Footnote | 14 |
| Kicker | 16 |
| Display | 76 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: Short topic or supported finding, dominant evidence, then interpretation and caveat
- **Composition tendency**: Broad open field with two unequal evidence regions or one wide table
- **Cross-page continuity**: Shared title and footer axes, gray rules, stable numeric precision
- **Spacing posture**: Open explanation pages alternating with dense comparisons
- **Spacing anchors**: page margin 64px, block gap 28px, column gutter 48px, corner radius 0px, body leading 36px

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eda_full_balance.png | 1577 × 654 | 2.4113 | Rule-match and other-transaction comparison by type | Evidence figure | Wide complete figure beside a fraud-coverage annotation | no-crop | user | Existing | Rendered unchanged from image/交易金额与余额行为/full_balance_rule_by_type.pdf; full labels and log-scale axes retained | embedded | local |
| eda_time_coverage.png | 723 x 904 | 0.7998 | Elapsed time coverage | Evidence figure | EDA page 05 | no-crop | user | Existing | Original time-distribution PDF | embedded | local |

## IX. Content Outline

#### Slide 01 - Transaction Fraud Detection
- **Audience move**: Unfamiliar with the challenge → oriented to the project and its contribution
- **Relationships**: none
- **Composition**: Oversized title with compact outcome subtitle
- **Title**: Transaction Fraud Detection
- **Core message**: A complete, evidence-led tabular ML case study
- **Content**: Project title; EDA, feature ablation and validation; Zhaobo HU

#### Slide 02 - Rare fraud, usable records
- **Audience move**: Unknown data quality → understand scale, rarity and usable inputs
- **Relationships**: contrast between clean records and difficult evaluation
- **Title**: Rare fraud, usable records
- **Core message**: Basic record quality is strong, but extreme imbalance governs evaluation
- **Content**: 3,144,097 transactions; 357 fraud; 0.0114%; no missing, duplicate or constant raw fields; pre-approval target

#### Slide 03 - The strongest observed relationship
- **Audience move**: Raw variables → understand amount/balance interaction
- **Relationships**: link between amount/balance proximity and observed fraud labels
- **Title**: The strongest observed relationship
- **Core message**: Positive-balance full-amount matches cover 346 of 357 fraud
- **Content**: A amount, B starting balance; B>0 and abs(A-B)<=0.01; 346/346 rule matches fraud; 11 fraud outside rule; full-file EDA caveat
- **Images**: eda_full_balance.png, complete source figure with native rule and coverage annotations

#### Slide 04 - Useful context and limited signals
- **Audience move**: Treat every association equally → distinguish type context from sparse historical evidence
- **Relationships**: contrast between observed category association and limited standalone detection
- **Title**: Useful context and limited signals
- **Core message**: Transaction type helps context, while flag and pair history have limited coverage
- **Content**: CASH_OUT 250/1,320,196; TRANSFER 107/337,569; system flag finds 2/357; prior same-pair count nonzero in 0/357 fraud; no universal feature sensitivity claim
- **Visualization**: type-evidence: transaction type, transaction count, fraud count, fraud rate
- **Native-ready**: type-evidence=yes

#### Slide 05 - Temporal coverage limits the evidence
- **Audience move**: Assume IID future behavior → understand restricted class coverage
- **Relationships**: order of early, mixed-class and fraud-only periods
- **Title**: Temporal coverage limits the evidence
- **Core message**: Label coverage makes random validation insufficient for future claims
- **Content**: No observed fraud before 635; mixed classes 635–718; only fraud 719–743; elapsed hours have no calendar origin

#### Slide 06 - Four feature groups isolate the additions
- **Audience move**: Candidate features → controlled feature-group ablation
- **Relationships**: membership of raw, extra and graph inputs in four configurations
- **Title**: Four feature groups isolate the additions
- **Core message**: Same estimator settings isolate the contribution of feature groups
- **Content**: 10 raw, 23 raw+extra, 22 raw+graph, 35 combined; extra ratios, differences, indicators; graph 12 strictly earlier-hour statistics
- **Composition**: Four assembly diagrams combine labelled raw, extra and graph blocks. Fixed block widths represent groups, not feature counts.

#### Slide 07 - What goes into Raw 10 and Extra 13?
- **Audience move**: Understand feature membership and validation boundaries
- **Relationships**: Feature composition and chronological order
- **Title**: What goes into Raw 10 and Extra 13?
- **Core message**: Explicit inputs and evidence boundaries support interpretation
- **Content**: See source SVG and submitted feature definitions; chronological cutoffs 659 / 680 / 718.

#### Slide 08 - Graph 12: the history behind a transaction
- **Audience move**: Understand feature membership and validation boundaries
- **Relationships**: Feature composition and chronological order
- **Title**: Graph 12: the history behind a transaction
- **Core message**: Explicit inputs and evidence boundaries support interpretation
- **Content**: See source SVG and submitted feature definitions; chronological cutoffs 659 / 680 / 718.

#### Slide 09 - Main experiment: stratified 60/20/20
- **Audience move**: Unclear splitting method → distinguish model fitting, calibration and holdout
- **Relationships**: order and distinct responsibilities of fit, calibration and holdout
- **Title**: Main experiment: stratified 60/20/20
- **Core message**: Two StratifiedShuffleSplit calls preserve rare-positive representation
- **Content**: Fit 1,886,457/214; calibration 628,820/72; holdout 628,820/71; seed 42; stratification preserves proportions, not temporal/entity independence
- **Composition**: Training file branches to 80% development and 20% holdout. Development branches to fit and calibration. Labels retain all original row and fraud counts. Geometry is schematic.

#### Slide 10 - Model selection and threshold selection
- **Audience move**: Mix model selection with evaluation → understand decision boundaries
- **Relationships**: order of candidate comparison, threshold choice and holdout scoring
- **Title**: Model selection and threshold selection
- **Core message**: Calibration chooses the model and cutoff before submission inference
- **Content**: Fit each candidate; choose F1 cutoff on calibration; rank by AP then F1 then fewer features; freeze; internal holdout; scores are not calibrated probabilities; repeated calibration use caveat

#### Slide 11 - Three baselines, one preprocessing pipeline
- **Audience move**: Feature groups → comparable model experiments
- **Relationships**: contrast among three estimators with shared preprocessing
- **Title**: Three baselines, one preprocessing pipeline
- **Core message**: Fixed configurations compare simple linear and boosted tree models
- **Content**: Balanced Logistic Regression; XGBoost; LightGBM; numeric median and scaling; type imputation and one-hot; fit partition only; no hyperparameter search
- **Composition**: Numeric and categorical preprocessing lanes merge into a feature matrix, which branches to the three estimators.

#### Slide 12 - Amount/balance features drive the gain
- **Audience move**: Model names → understand measured ablation results
- **Relationships**: contrast of four groups and three classifiers
- **Title**: Amount/balance features drive the gain
- **Core message**: Extra features dominate the observed improvement, graph adds no gain to the selected combination
- **Content**: All 12 holdout AP values rounded to four decimals; calibration AP selects candidates; LightGBM findings apply to tested settings
- **Visualization**: ap-results: four feature groups by three estimators, holdout AP
- **Native-ready**: ap-results=yes

#### Slide 13 - The selected operating point
- **Audience move**: Ranking scores → understand actual alert counts
- **Relationships**: contrast between actual and predicted labels
- **Title**: The selected operating point
- **Core message**: Selected 23-feature logistic model detects 71/71 with zero false alerts internally
- **Content**: Confusion counts; threshold 0.999296488; rule baseline finds 70/71; small positive support and discovery exposure
- **Visualization**: confusion-counts: actual-by-predicted confusion table
- **Native-ready**: confusion-counts=yes

#### Slide 14 - Ranking metrics under extreme imbalance
- **Audience move**: Perfect accuracy intuition → AP and ROC-AUC ranking interpretation
- **Relationships**: contrast of accuracy, AP and ROC-AUC
- **Title**: Ranking metrics under extreme imbalance
- **Core message**: AP evaluates fraud ranking while accuracy conceals missed fraud
- **Content**: All-negative accuracy 99.9886%, recall 0; AP weighted precision over recall increments; ROC-AUC pairwise ranking with half-credit ties; continuous scores; same-partition prevalence baseline
- **Mathematical content**: AP=\sum_n (R_n-R_{n-1})P_n

#### Slide 15 - Metrics at a fixed threshold
- **Audience move**: Ranking quality → operational alert interpretation
- **Relationships**: link between confusion counts and operational metrics
- **Title**: Metrics at a fixed threshold
- **Core message**: Precision measures alert quality, recall fraud coverage, F1 balances both
- **Content**: Precision=TP/(TP+FP); Recall=TP/(TP+FN); F1=2TP/(2TP+FP+FN); false alerts per 100k; alerts; actual holdout counts; no supplied cost or alert budget

#### Slide 16 - Three-fold stratified validation
- **Audience move**: Single-split confidence → understand repeated unseen-fold evaluation
- **Relationships**: membership of validation and training rows across three outer folds
- **Title**: Three-fold stratified validation
- **Core message**: Every outer fold evaluates rows absent from its fitting and threshold selection
- **Content**: StratifiedKFold K=3, shuffle seed42; inner 80/20 threshold selection; refit complete outer training; test untouched outer fold; only raw and raw+extra, all estimators
- **Composition**: Three partition strips rotate a white validation block among gray training blocks. A lower process strip explains inner calibration and refitting.

#### Slide 17 - Feature gains persist across folds
- **Audience move**: Validation mechanism → actual stability evidence and its limit
- **Relationships**: contrast of raw and extra within estimator
- **Title**: Feature gains persist across folds
- **Core message**: Random-fold stability supports the engineered features beyond a single split
- **Content**: LR raw AP .25204 F1 .34459; LR extra AP .99998 ± .00004 F1 .99149 ± .00737; XGB raw AP .76539 F1 .66297; XGB extra AP .99926 ± .00129 F1 .98997 ± .01387; no claim that randomness or leakage is eliminated
- **Visualization**: ap-gain-chart: horizontal bars compare mean AP for LR raw, LR + extra, XGB raw and XGB + extra on a shared 0–1 scale. Companion summary retains AP SD and mean F1 for engineered models.
- **Native-ready**: ap-gain-chart=yes

#### Slide 18 - Supplementary: chronological split
- **Audience move**: Understand feature membership and validation boundaries
- **Relationships**: Feature composition and chronological order
- **Title**: Supplementary: chronological split
- **Core message**: Explicit inputs and evidence boundaries support interpretation
- **Content**: See source SVG and submitted feature definitions; chronological cutoffs 659 / 680 / 718.

#### Slide 19 - Later-period results: fewer false alerts
- **Audience move**: From temporal partition to measured model comparison
- **Relationships**: Compare three feature variants on the same later holdout
- **Title**: Later-period results: fewer false alerts
- **Core message**: Extra retains 64 detections while removing 8,452 false alerts
- **Content**: Raw versus Extra versus Extra without hour and flag; AP, fraud found, FP, F1; short post-hoc caution
- **Visualization**: temporal-results table
- **Native-ready**: temporal-results=yes

#### Slide 20 - Final inference and practical next step
- **Audience move**: Experimental result → final deliverable and sensible next validation
- **Relationships**: order of full-training refit and external inference
- **Title**: Final inference and practical next step
- **Core message**: Submit the simple selected model, then validate it prospectively
- **Content**: Full-training refit; frozen threshold; 2,247,484 external rows, 829 predicted positives; preserve all original rows and columns; no external labels or measured external accuracy; next untouched future evaluation

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under notes/
- **Content**: English presenter notes explain evidence, calculations and caveats. Cite repository-relative code, CSV and figure sources. Keep internal production commentary out.
- **Total duration**: 12–15 minutes
- **Notes style**: Conversational technical interview
- **Presentation purpose**: Explain the complete fraud detection project and the strength and limits of its evidence

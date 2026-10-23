<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: en-US
- audience: Data science recruitment panel
- objective: Explain the complete project and let reviewers assess feature evidence, metrics and validation limits.
- core_message: Amount/balance features drive the observed gain, while discovery exposure and time coverage limit future claims.
- consumption_mode: presentation

## mode
- mode: custom
- mode_behavior: Follow EDA, feature ablation, baselines and results, then explain metrics, splitting and stability with clear caveats.

## visual_style
- visual_style: custom
- visual_style_behavior: Black field, white Arial, open margins, source EDA evidence figures, editable data charts and relationship diagrams, with compact tables retained for exact lookup.

## colors
- background: #000000
- secondary_bg: #111111
- primary: #F5F5F5
- accent: #FFFFFF
- secondary_accent: #D0D2D6
- body_text: #F5F5F5
- secondary_text: #A7ABB2
- divider: #36383D

## typography
- font_family: Arial
- title_family: Arial
- body_family: Arial
- body: 26
- title: 44
- subtitle: 30
- annotation: 20
- table: 23
- footnote: 14
- kicker: 16
- display: 76

## icons
- library: none
- inventory: none

## page_rhythm
- P01: breathing
- P02: anchor
- P03: anchor
- P04: dense
- P05: breathing
- P06: dense
- P07: anchor
- P08: dense
- P09: anchor
- P10: anchor
- P11: dense
- P12: dense
- P13: breathing
- P14: dense
- P15: dense
- P16: anchor
- P17: breathing

## pptx_structure
- mode: flat

## images
- time-coverage: images/eda_time_coverage.png | source=user | crop=no-crop
- full-balance: images/eda_full_balance.png | source=user | crop=no-crop

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters

## Revision: feature detail and experiment ordering
20 pages. Main stratified experiment precedes results. K-fold and chronological checks are supplementary. Two feature inventory pages added. Original time-coverage EDA image restored on page 05.

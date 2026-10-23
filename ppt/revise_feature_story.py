from pathlib import Path
import re

w = Path('ppt/fraud_detection_20261023')
d = w / 'svg_output'
old = {p.name[:2]: (p.name[3:], p.read_text(encoding='utf-8')) for p in d.glob('*.svg')}
notes = {p.name[:2]: p.read_text(encoding='utf-8') for p in (w/'notes').glob('[0-9][0-9]_*.md')}

def page(title, body, source):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" font-family="Arial" data-pptx-page-role="content">
<rect id="background" data-pptx-role="background" width="1280" height="720" fill="#000000"/>
<g id="title" data-pptx-bounds="64 48 1152 94" fill="#F5F5F5"><text x="64" y="70" font-size="16" fill="#A7ABB2">02 FEATURE ENGINEERING</text><text x="64" y="124" font-size="42">{title}</text></g>
{body}
<text id="source" data-pptx-role="footer" x="64" y="680" font-size="14" fill="#A7ABB2">{source}</text><text id="page" data-pptx-role="page-number" x="1216" y="680" text-anchor="end" font-size="14" fill="#A7ABB2">00 / 20</text></svg>'''

tabular = page('What goes into Raw 10 and Extra 13?', '''
<g id="raw" data-pptx-bounds="64 180 510 446" fill="#F5F5F5">
<text x="64" y="209" font-size="29">Raw 10: transaction context</text>
<text x="64" y="257" font-size="23">Elapsed hour; transaction type; amount</text>
<text x="64" y="299" font-size="23">Initiator + recipient starting balances</text>
<text x="64" y="341" font-size="23">Existing system fraud flag</text>
<text x="64" y="383" font-size="23">C / M prefix flags for both endpoints (4)</text>
<line x1="64" y1="416" x2="574" y2="416" stroke="#36383D"/>
<text x="64" y="459" font-size="22">Pre-approval inputs only</text>
<text x="64" y="498" font-size="20" fill="#A7ABB2">Account IDs define history, not model columns.</text>
<text x="64" y="536" font-size="20" fill="#A7ABB2">Post-transaction balances are excluded.</text>
<text x="64" y="599" font-size="20" fill="#A7ABB2">Counts refer to inputs before one-hot encoding.</text>
</g>
<g id="extra" data-pptx-bounds="640 180 576 446" fill="#F5F5F5">
<text x="640" y="209" font-size="29">Extra 13: amount versus balance</text>
<text x="640" y="257" font-size="23">Log amount (1); zero starting balances (2)</text>
<text x="640" y="299" font-size="23">Amount / each starting balance (2)</text>
<text x="640" y="341" font-size="23">Signed + absolute initiator difference (2)</text>
<text x="640" y="383" font-size="23">Exceeds balance; full-balance match (2)</text>
<text x="640" y="425" font-size="23">Zero amount; zero amount + balance (2)</text>
<text x="640" y="467" font-size="23">Full-balance x TRANSFER / CASH_OUT (2)</text>
<text x="640" y="530" font-size="23">Full balance: B &gt; 0 and |A - B| &lt;= 0.01</text>
<text x="640" y="568" font-size="20" fill="#A7ABB2">A = amount; B = initiator starting balance.</text>
<text x="640" y="606" font-size="20" fill="#A7ABB2">Ratios use positive denominators; otherwise NaN.</text>
</g>''', 'Source: submitted train_ablation.py, RAW_COLUMNS / EXTRA_COLUMNS and add_extra_features().')

graph = page('Graph 12: the history behind a transaction', '''
<g id="graph-example" data-pptx-bounds="64 177 500 275" fill="#F5F5F5">
<text x="64" y="205" font-size="21" fill="#A7ABB2">Current query: A sends to B at hour h</text>
<circle cx="198" cy="318" r="44" fill="#111111" stroke="#F5F5F5" stroke-width="2"/><circle cx="438" cy="318" r="44" fill="#111111" stroke="#F5F5F5" stroke-width="2"/>
<text x="198" y="328" text-anchor="middle" font-size="30">A</text><text x="438" y="328" text-anchor="middle" font-size="30">B</text>
<path d="M242 306 H389" fill="none" stroke="#F5F5F5" stroke-width="3"/><path d="M389 306 L377 299 L377 313 Z" fill="#F5F5F5"/>
<text x="318" y="284" text-anchor="middle" font-size="20">current edge</text>
<path d="M394 343 Q318 412 242 343" fill="none" stroke="#A7ABB2" stroke-width="2"/><text x="318" y="424" text-anchor="middle" font-size="19" fill="#A7ABB2">earlier B-to-A edges: reciprocal count</text>
</g>
<g id="history-rules" data-pptx-bounds="64 470 510 155" fill="#F5F5F5"><text x="64" y="498" font-size="24">Only edges from hours strictly before h</text><text x="64" y="538" font-size="20" fill="#A7ABB2">Same-hour transactions never update each other.</text><text x="64" y="576" font-size="20" fill="#A7ABB2">No labels or after-balances enter these features.</text><text x="64" y="614" font-size="20" fill="#A7ABB2">Online use assumes available earlier unlabeled history.</text></g>
<g id="graph-inventory" data-pptx-bounds="640 177 576 448" fill="#F5F5F5">
<text x="640" y="209" font-size="27">Counts and counterparties (5)</text>
<text x="640" y="251" font-size="22">A: outgoing count + distinct recipients</text><text x="640" y="287" font-size="22">B: incoming count + distinct initiators</text><text x="640" y="323" font-size="22">A-to-B: prior transaction count</text>
<text x="640" y="379" font-size="27">Historical amounts (6)</text><text x="640" y="419" font-size="22">Sum / mean / max: A outgoing, B incoming</text>
<text x="640" y="475" font-size="27">Reciprocity (1)</text><text x="640" y="515" font-size="22">Prior B-to-A count for the current A-to-B edge</text>
<text x="640" y="573" font-size="20" fill="#A7ABB2">GFP-inspired local statistics, not a full motif miner.</text><text x="640" y="610" font-size="20" fill="#A7ABB2">No incremental AP gain over Extra in this ablation.</text></g>''', 'Source: submitted train_ablation.py GRAPH_COLUMNS; graph_features.py; gfp_additional_features.py; model_metrics.csv.')

eda = (w/'backup/20261023_100344/svg_output/05_temporal_coverage.svg').read_text(encoding='utf-8')
roster = [('old',k) for k in ['01','02','03','04']] + [('eda','05'),('old','06'),('tab','00'),('graph','00')] + [('old',k) for k in ['12','13','07','08','09','10','11','14','15','05','16','17']]
for p in d.glob('*.svg'): p.unlink()
for p in (w/'notes').glob('[0-9][0-9]_*.md'): p.unlink()
for p in (w/'svg_final').glob('*.svg'): p.unlink()
note_sections=[]
for i,(kind,key) in enumerate(roster,1):
    if kind=='old': name,s=old[key]; note=notes.get(key,'')
    elif kind=='eda': name,s='temporal_coverage.svg',eda; note='EDA: fraud coverage starts at hour 635; legitimate records end at 718. This motivates the later supplementary chronological check.'
    elif kind=='tab': name,s='tabular_feature_inventory.svg',tabular; note='Complete semantic inventory: Raw 10 plus Extra 13. Both zero-balance flags refer to starting balances. Full-balance type interactions multiply the matching rule by TRANSFER and CASH_OUT indicators. All 13 additions are defined without target labels; hypothesis discovery nevertheless used full-training EDA.'
    else: name,s='graph_feature_inventory.svg',graph; note='Graph 12 is five count features, six amount statistics and one reverse-edge count. In this implementation degree means transaction count, while fan means number of distinct counterparties. Reciprocal cycle count is earlier reverse-edge frequency, not an arbitrary cycle search. Features use hours strictly less than the current hour. Graph features are label-free but the random protocol assumes available earlier unlabeled history across partitions. Their value is assessed by ablation; they did not improve the selected Extra model.'
    if kind=='old' and key in ['14','15','05','16']:
        s=s.replace('06  VALIDATION','07  SUPPLEMENTARY CHECKS').replace('06 VALIDATION','07 SUPPLEMENTARY CHECKS')
    if kind=='old' and key=='05': s=s.replace('Chronological sensitivity split','Supplementary: chronological split')
    s=re.sub(r'\d{2} / 17|00 / 20',f'{i:02d} / 20',s)
    stem=f'{i:02d}_{name[:-4]}'
    (d/(stem+'.svg')).write_text(s,encoding='utf-8')
    (w/'notes'/(stem+'.md')).write_text(note,encoding='utf-8')
    note_sections.append(f'## {stem}\n\n{note}\n')
(w/'notes/total.md').write_text('\n'.join(note_sections),encoding='utf-8')
print('Reordered 20 slides; added complete tabular and graph inventory pages.')

"""Slot errors on dev2 (truth for scoring only). usage: slots.py [run root] [-v]"""
import json, sys, math
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[3]
run = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else ROOT/'out/bench/dev2'
V = '-v' in sys.argv
J = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
bench = json.loads((run/'bench.json').read_text())
cat = Counter(); dist = Counter(); rows=[]
for c in bench['clips']:
    seed = c['clip']
    clip = ROOT/f'data/synth/bench/dev2/clip_{seed}'
    lay = json.loads((clip/'layout.json').read_text())
    S = {s['id']: s for s in lay['slots']}
    raw = J(run/f'clip_{seed}/pipeline/shelf_events.jsonl')
    acts = J(run/f'clip_{seed}/pipeline/store_shelf_events.jsonl')
    for r in c['picks']:
        st, d = r['stages'], r['detail']
        if not st['shelf_event_emitted'] or st['right_slot']:
            continue
        ts, es = S[r['slot']], S.get(d['event_slot'])
        dd = math.dist(ts['face'], es['face']) if es else None
        same_sku = es and es.get('skuId') == ts.get('skuId')
        near = [e for e in raw if e['kind']=='take' and abs(e['t']-r['t'])<=3.5 and e.get('point_3d') and math.dist(e['point_3d'], ts['face'])<=0.6]
        exact = [e for e in near if e['slot_id']==r['slot']]
        cand = [e for e in near if any(s==r['slot'] for s,_ in e.get('slots') or [])]
        act_true = [a for a in acts if a['kind']=='take' and abs(a['t']-r['t'])<=3.5 and a['slot_id']==r['slot']]
        k = ('act at true slot exists (pairing)' if act_true else 'reading at true slot, lost in fusion' if exact else 'true slot is a candidate only' if cand else 'readings near, none names it' if near else 'no reading near')
        cat[k]+=1
        dist['same fixture row' if es and es['id'].rsplit('-',1)[0]==ts['id'].rsplit('-',1)[0] else 'same fixture other shelf' if es and es.get('fixtureId')==ts.get('fixtureId') else 'other fixture' if es else 'no slot'] += 1
        rows.append((seed, r['shopper'], r['t'], r['slot'], d['event_slot'], round(dd,2) if dd is not None else None, same_sku, k, [(e['camera_id'], e['t'], e['slot_id'], e['source'], e.get('cue'), [(s,round(w,2)) for s,w in (e.get('slots') or [])][:3]) for e in near]))
print(sum(cat.values()), dict(cat)); print(dict(dist))
import statistics
ds=[x[5] for x in rows if x[5] is not None]; print('dist median', statistics.median(ds), 'under 0.12', sum(d<=0.12 for d in ds), 'under 0.25', sum(d<=0.25 for d in ds), 'same sku', sum(bool(x[6]) for x in rows))
if V:
    for x in rows:
        print(*x[:8]);
        for n in x[8]: print('     ', n)

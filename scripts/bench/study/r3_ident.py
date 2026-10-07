"""Where identities change person on dev2 (truth for scoring only). usage: ident.py [run root]"""
import json, sys, math
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
from bree.sim.bench import _iou, clip_dirs
run = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'out/bench/dev2'
J = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
tot = Counter(); sw = []
for clip in clip_dirs('dev2'):
    pipe = run/clip.name/'pipeline'
    meta = json.loads((clip/'clip.json').read_text()); fps = meta['fps']
    tf = {(r['camera'], r['frame']): r for r in J(clip/'truth'/'frames.jsonl')}
    pos = {r['frame']: {s['shopper']: s for s in r['shoppers']} for r in J(clip/'truth'/'tracks.jsonl')}
    sh = {s['shopper']: s for s in json.loads((clip/'truth'/'shoppers.json').read_text())}
    seen = defaultdict(list)
    for cam in meta['cameras']:
        f = pipe/f'frames_{cam}.jsonl'
        if not f.exists(): continue
        for row in J(f):
            fr = round(row['t']*fps); gt = (tf.get((cam, fr)) or {}).get('persons', [])
            for p in row['persons']:
                if p.get('gid') is None: continue
                best = max(gt, key=lambda g: _iou(g['bbox'], p['bbox']), default=None)
                if best and _iou(best['bbox'], p['bbox']) >= 0.3:
                    seen[p['gid']].append((fr, best['shopper']))
    for gid, obs in seen.items():
        byf = defaultdict(Counter)
        for fr, s in obs: byf[fr][s] += 1
        frs = sorted(byf); lab = {}
        for fr in frs:      # majority over +-5 frames
            c = Counter()
            for g in range(fr-5, fr+6): c.update(byf.get(g, {}))
            lab[fr] = c.most_common(1)[0][0]
        runs = []
        for fr in frs:
            if runs and runs[-1][0] == lab[fr]: runs[-1][2] = fr
            else: runs.append([lab[fr], fr, fr])
        runs = [r for r in runs if r[2]-r[1] >= 10]      # a second or more on that person
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]: merged[-1][2] = r[2]
            else: merged.append(list(r))
        tot['ids'] += 1; tot['ids that change person'] += len(merged) > 1
        for a, b in zip(merged, merged[1:]):
            fr = b[1]; pa, pb = (pos.get(a[2], {}).get(a[0]), pos.get(b[1], {}).get(b[0]))
            pa2 = pos.get(fr, {}).get(a[0])
            d = math.dist((pa2['x'], pa2['z']), (pb['x'], pb['z'])) if pa2 and pb and 'x' in pb else None
            sw.append(dict(clip=clip.name, gid=gid, t=fr/fps, a=a[0], b=b[0], gap=(b[1]-a[2])/fps, d=d, a_present=pa2 is not None,
                           group=bool(sh.get(a[0], {}).get('group')) and sh.get(a[0], {}).get('group') == sh.get(b[0], {}).get('group'), staff='STAFF' in a[0]+b[0] or 'clerk' in (a[0], b[0]),
                           where=(round(pb['x'], 1), round(pb['z'], 1)) if pb and 'x' in pb else None))
print(dict(tot), 'switches', len(sw))
c = Counter()
for s in sw:
    c['A gone from the store' if not s['a_present'] else 'A and B within 0.5 m' if s['d'] is not None and s['d'] <= 0.5 else 'within 1.0 m' if s['d'] is not None and s['d'] <= 1.0 else 'within 2 m' if s['d'] is not None and s['d']<=2 else 'further']+=1
    c['same group'] += s['group']; c['staff or clerk involved'] += s['staff']; c['gap > 1 s (hand-back)'] += s['gap'] > 1.0; c['gap <= 0.3 s (track kept, person changed)'] += s['gap'] <= 0.3
print(dict(c))
json.dump(sw, open(str(ROOT / 'out' / 'bench' / 'r3' / 'switch.json'), 'w'))
for s in sw[:40]: print(s)

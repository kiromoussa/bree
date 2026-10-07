"""Tracker only, on stored person boxes, scored on the floor against the true positions (truth for scoring only).
usage: idfast.py '<FloorConfig json>' [split] [-m: list meetings]"""
import json, sys, math
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
from bree.shelf import store
from bree.events.shelf import track_people
from bree.track.floor import FloorConfig
from bree.track.people import PEOPLE_KINDS
from bree.sim.bench import load_calibration, clip_dirs
J = store._jsonl
cfg = FloorConfig(**json.loads(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1].startswith('{') else None
split = next((a for a in sys.argv[2:] if not a.startswith('-')), 'dev2')
SRC = ROOT/'out/bench'/split
tot = Counter(); meet = Counter(); rows = []; cause = Counter(); sw = []
for clip in clip_dirs(split):
    pipe = SRC/clip.name/'pipeline'
    if not (pipe/'shelf_events.jsonl').exists(): continue
    layout, meta = json.loads((clip/'layout.json').read_text()), json.loads((clip/'clip.json').read_text())
    kind = {c['id']: c['kind'] for c in json.loads((clip/'calibration.json').read_text())['cameras']}
    cams = {c: v for c, v in load_calibration(clip).items() if c in meta['cameras'] and kind[c] in PEOPLE_KINDS}
    boxes = {c: J(pipe/f'people_{c}.jsonl') for c in cams}
    tracker, ids = track_people(cams, layout, boxes, float(meta['fps']), cfg)
    pos = {r['frame']: {s['shopper']: (s['x'], s['z']) for s in r['shoppers']} for r in J(clip/'truth'/'tracks.jsonl')}
    def who(t, x, z):
        P = pos.get(round(t*10)) or {}
        d, n = min(((math.dist((x, z), p), n) for n, p in P.items()), default=(9, None))
        return n if d <= 0.6 else None
    lab = {}; per = defaultdict(list)
    for k in tracker.people():
        if k.staff: continue
        seq = [(t, who(t, x, z)) for t, x, z in k.path]
        lab[k.id] = seq
        c = Counter(n for _, n in seq if n)
        if sum(c.values()) < 5: continue
        tot['ids'] += 1
        main, nmain = c.most_common(1)[0]
        tot['ids on two people'] += nmain < 0.9*sum(c.values())
        per[main].append(k.id)
        # switches: runs of 1 s or more
        runs = []
        for t, n in seq:
            if n is None: continue
            if runs and runs[-1][0] == n: runs[-1][2] = t
            else: runs.append([n, t, t])
        runs = [r for r in runs if r[2]-r[1] >= 1.0]
        tot['switches'] += sum(a[0] != b[0] for a, b in zip(runs, runs[1:]))
        import re
        for a, b in zip(runs, runs[1:]):
            if a[0] == b[0]: continue
            ts = b[1]
            back = [l for l in tracker.log if re.match(rf'\s*[0-9.]+s id {k.id} seen again', l) and a[2] - 0.5 <= float(l.split('s')[0]) <= ts + 2.0]
            ms = [m for m in tracker.settled if k.id in m['ids'] and m['t0'] - 1 <= ts and a[2] - 1 <= m['t']]
            op = [m for m in tracker.meetings if k.id in (m['a'].id, m['b'].id) and m['t0'] - 1 <= ts]
            cause[('handed back to another person' if back else 'track kept, person changed') + ': ' + ('meeting ' + ms[-1]['said'] if ms else 'meeting open at the end' if op else 'no meeting')] += 1
            sw.append((clip.name, k.id, a[0], b[0], round(a[2], 1), round(ts, 1), back[-1].strip()[:150] if back else '', [(m['t0'], m['t'], m['ids'], m['said']) for m in ms]))
    names = {n for P in pos.values() for n in P}
    tot['people'] += len(names); tot['ids of people'] += sum(len(v) for v in per.values()); tot['people with no id'] += len(names - set(per))
    def true_of(i, t0, t1):
        c = Counter(n for t, n in lab.get(i, []) if t0 <= t <= t1 and n)
        return c.most_common(1)[0][0] if c else None
    for m in tracker.settled:
        a, b = m['ids']       # ids as they were when it was settled (before a swap is applied the ids are the old ones)
        # truth: who each track body followed before the meeting and after it. After a put-right swap the ids are exchanged from t0, so read the final labels:
        fin_before = (true_of(a, m['t0']-3, m['t0']-0.3), true_of(b, m['t0']-3, m['t0']-0.3))
        fin_after = (true_of(a, m['t']-0.5, m['t']+3), true_of(b, m['t']-0.5, m['t']+3))
        if None in fin_before or None in fin_after or fin_before[0] == fin_before[1]:
            meet[(m.get('sides', 2), m['said'], 'truth unclear')] += 1; continue
        ok = fin_before == fin_after
        meet[(m.get('sides', 2), m['said'], 'ids right afterwards' if ok else 'ids crossed afterwards' if fin_before == fin_after[::-1] else 'other')] += 1
        rows.append((clip.name, m['t0'], m['t'], m['ids'], m['said'], round(m.get('straight', -1), 2), round(m.get('crossed', -1), 2), ok))
    for k, v in tracker.counts.items(): tot['count ' + k] += v
print(json.dumps(sys.argv[1] if cfg else 'default'), split)
print('ids per person', round(tot['ids of people']/max(tot['people'], 1), 3), ' ids on two people', tot['ids on two people'], 'of', tot['ids'], round(tot['ids on two people']/max(tot['ids'],1), 3), ' switches', tot['switches'], ' people with no id', tot['people with no id'])
print({k[6:]: v for k, v in tot.items() if k.startswith('count ')})
for k in sorted(meet): print('  meeting', k, meet[k])
if '-m' in sys.argv:
    for r in rows: print(r)
for k in sorted(cause): print('  switch', k, cause[k])
if '-s' in sys.argv:
    for r in sw: print(r)

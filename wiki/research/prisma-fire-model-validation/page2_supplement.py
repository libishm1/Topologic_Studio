"""The FMV review's second harvest page (PROTOCOL.md section 3).

OpenAlex now needs an API key to page past the first 200 records with a
cursor, so the skill's harvest_openalex.py stops at 200 for queries with more
hits. This fetches records 201 to `cap` (400 here) of each such query with
plain page numbers, which need no key, by the same filter, ranking and
fields, and merges them into the harvest: each record keeps the queries that
found it and its rank in each; the search log's `retrieved` is updated and
the pages fetched are recorded.

    python -P page2_supplement.py queries.json work/harvest
"""
import json
import os
import re
import sys
import time

import requests

SEL = 'id,doi,title,display_name,publication_year,publication_date,type,primary_location,open_access,best_oa_location,cited_by_count,authorships,abstract_inverted_index,language,topics,keywords,fwci,biblio'


def inv2abs(ii):
    if not ii:
        return ''
    pos = {}
    for w, ps in ii.items():
        for p in ps:
            pos[p] = w
    return ' '.join(pos[k] for k in sorted(pos))


cfg = json.load(open(sys.argv[1], encoding='utf-8'))
out = sys.argv[2]
blocks = cfg.get('blocks') or {}
expand = lambda s: re.sub(r'\{(\w+)\}', lambda m: blocks.get(m.group(1), m.group(0)), s)
recs = {r['id']: r for r in json.load(open(os.path.join(out, 'openalex_raw.json'), encoding='utf-8'))}
log = json.load(open(os.path.join(out, 'search_log.json'), encoding='utf-8'))
d0, d1, types = cfg['date_from'], cfg['date_to'], cfg.get('types')
for qid, q in cfg['queries'].items():
    q = q if isinstance(q, dict) else {'query': q}
    cap = int(q.get('cap', cfg.get('default_cap', 600)))
    entry = log.get(qid) or {}
    total, got = entry.get('total_hits') or 0, entry.get('retrieved') or 0
    page = got // 200 + 1
    while got < min(cap, total) and got % 200 == 0:
        flt = f'title_and_abstract.search:{expand(q["query"])},from_publication_date:{d0},to_publication_date:{d1}' + (f',type:{types}' if types else '')
        r = None
        for attempt in range(4):
            r = requests.get('https://api.openalex.org/works', params={'filter': flt, 'per_page': 200, 'page': page, 'select': SEL, 'sort': 'relevance_score:desc'}, timeout=60)
            if r.status_code == 200:
                break
            print(qid, 'HTTP', r.status_code, r.text[:160])
            time.sleep(3)
        if r is None or r.status_code != 200:
            break
        results = r.json()['results']
        if not results:
            break
        for w in results:
            got += 1
            if got > cap:
                break
            wid = w['id']
            if wid not in recs:
                w['abstract'] = inv2abs(w.pop('abstract_inverted_index', None))
                w['queries'] = []
                w['rel_rank'] = {}
                recs[wid] = w
            if qid not in recs[wid]['queries']:
                recs[wid]['queries'].append(qid)
            recs[wid]['rel_rank'][qid] = got
        entry.setdefault('pages_by_number', []).append(page)
        page += 1
        time.sleep(0.5)
    entry['retrieved'] = min(got, cap)
    log[qid] = entry
    print(f'{qid}: total={total} retrieved={entry["retrieved"]} unique_so_far={len(recs)}', flush=True)
json.dump(list(recs.values()), open(os.path.join(out, 'openalex_raw.json'), 'w', encoding='utf-8'))
json.dump(log, open(os.path.join(out, 'search_log.json'), 'w', encoding='utf-8'), indent=1)
print('UNIQUE', len(recs))

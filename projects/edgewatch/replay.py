"""Replay a synthetic device's durable local outbox; retain failed records."""
import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def flush(path, url, key):
    records = path.read_text().splitlines()
    sent = 0
    for line in records:
        request = urllib.request.Request(url + '/api/samples', line.encode(),
            {'Content-Type': 'application/json', 'X-API-Key': key})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status not in (200, 201):
                    break
        except OSError:
            break
        sent += 1
    temp = path.with_suffix('.tmp')
    temp.write_text('\n'.join(records[sent:]) + ('\n' if records[sent:] else ''))
    temp.replace(path)
    return sent


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--outbox', type=Path, default=Path('outbox.jsonl'))
    p.add_argument('--seed', action='store_true')
    a = p.parse_args()
    if a.seed:
        if a.outbox.exists():
            p.error('Outbox exists; drain or move it before seeding')
        now = datetime.now(timezone.utc).isoformat()
        a.outbox.write_text(''.join(json.dumps(dict(device='sim-esp32', seq=i, ts=now,
            temperature=72 if i == 8 else 24 + i / 10, voltage=3.0 if i == 9 else 3.8)) + '\n' for i in range(10)))
    print({'sent': flush(a.outbox, a.url, os.environ.get('EDGEWATCH_KEY', ''))})

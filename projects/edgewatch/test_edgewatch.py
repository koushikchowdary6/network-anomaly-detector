import copy
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import json
from pathlib import Path
from http.server import ThreadingHTTPServer
from app import connect, ingest, handler
from replay import flush

SAMPLE = dict(device='sim-1', seq=1, ts='2026-01-01T00:00:00+00:00', temperature=25, voltage=3.8)

class Tests(unittest.TestCase):
    def setUp(self):
        self.db = connect(':memory:')
    def tearDown(self):
        self.db.close()
    def test_replay_is_idempotent(self):
        self.assertTrue(ingest(self.db, SAMPLE)['accepted'])
        self.assertTrue(ingest(self.db, SAMPLE)['duplicate'])
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0], 1)
    def test_conflict(self):
        ingest(self.db, SAMPLE)
        with self.assertRaises(ValueError): ingest(self.db, dict(SAMPLE, voltage=4))
    def test_alerts(self):
        self.assertEqual(ingest(self.db, dict(SAMPLE, temperature=70, voltage=3))['alert'], 'HIGH_TEMPERATURE, LOW_VOLTAGE')
    def test_bad_measurements(self):
        for field, value in [('voltage', float('nan')), ('temperature', True), ('seq', -1), ('ts', '2026-01-01')]:
            with self.subTest(field=field), self.assertRaises(ValueError): ingest(self.db, dict(SAMPLE, **{field: value}))
    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'db'
            db=connect(path); ingest(db, SAMPLE); db.close()
            db=connect(path); self.assertTrue(ingest(db, SAMPLE)['duplicate']); db.close()
    def test_http_and_outbox(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'db'; connect(path).close()
            server=ThreadingHTTPServer(('127.0.0.1',0), handler(path,'x'*16))
            thread=threading.Thread(target=server.serve_forever); thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                outbox=Path(d)/'outbox'; outbox.write_text(json.dumps(SAMPLE)+'\n')
                self.assertEqual(flush(outbox,url,'wrong'),0)
                self.assertTrue(outbox.read_text())
                self.assertEqual(flush(outbox,url,'x'*16),1)
                self.assertEqual(outbox.read_text(),'')
                request=urllib.request.Request(url+'/api/samples',headers={'X-API-Key':'x'*16})
                with urllib.request.urlopen(request) as r: self.assertEqual(len(json.load(r)),1)
            finally:
                server.shutdown(); thread.join(); server.server_close()

if __name__ == '__main__': unittest.main()

"""Run from any folder: python app/app.py. Local inference; no training."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import argparse
import json
from model_loader import Models
ROOT = Path(__file__).resolve().parents[1]
MODELS = None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT/'frontend'), **kwargs)

    def send_json(self, status, value):
        body = json.dumps(value).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status': self.send_json(200, {'models':MODELS.status()})
        else: super().do_GET()

    def do_POST(self):
        if self.path != '/api/predict': self.send_json(404, {'error':'Unknown endpoint.'}); return
        # This service is intended for localhost. Reject cross-origin browser submissions.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            self.send_json(403, {'error':'Cross-origin requests are not accepted.'}); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 250000: raise ValueError('Request is empty or too large.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict): raise ValueError('Expected a JSON object.')
            subject, body = payload.get('subject',''), payload.get('body','')
            if not isinstance(subject,str) or not isinstance(body,str): raise ValueError('Subject and body must be text.')
            if not body.strip(): raise ValueError('Please enter email content.')
            if len(subject)>500 or len(body)>50000: raise ValueError('Email exceeds the input length limit.')
            # Preserve raw text for each saved vectorizer. Do not apply a new common normalizer.
            text = (subject.strip()+'\n'+body).strip() if subject.strip() else body
            self.send_json(200, {'results':MODELS.predict(text)})
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {'error':str(exc)})

    def log_message(self, fmt, *args):
        # Never log email text or request bodies.
        super().log_message(fmt, *args)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    MODELS=Models()
    print(json.dumps(MODELS.status(),indent=2),flush=True)
    print(f'Open http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

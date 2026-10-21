"""Read-only localhost viewer for a recorded CGB development run."""
import argparse
import gzip
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


class Viewer(BaseHTTPRequestHandler):
    def do_GET(self):
        query=urlsplit(self.path)
        route=query.path
        version=parse_qs(query.query).get('version',['current'])[0]
        if version not in self.server.scenes:
            self.send_error(404,'Unknown saved version')
            return
        if route in ('/','/index.html'):
            data=(Path(__file__).parent/'index.html').read_bytes()
            content_type='text/html; charset=utf-8'
        elif route=='/scene.json':
            data=self.server.scenes[version][0]
            content_type='application/json; charset=utf-8'
        elif route=='/versions':
            data=json.dumps(self.server.versions,ensure_ascii=False).encode('utf-8')
            content_type='application/json; charset=utf-8'
        elif route=='/calibration':
            data=self.server.calibration
            content_type='application/json; charset=utf-8'
        elif route=='/material':
            data=(Path(__file__).parent/'material.html').read_bytes()
            content_type='text/html; charset=utf-8'
        elif route=='/material_scene.json':
            if self.server.material_scene is None:
                self.send_error(404,'No physical material run attached')
                return
            data=self.server.material_scene
            content_type='application/json; charset=utf-8'
        elif route=='/health':
            data=b'{"status":"ready","read_only":true}'
            content_type='application/json'
        else:
            self.send_error(404)
            return
        compressed=route=='/scene.json' and 'gzip' in self.headers.get('Accept-Encoding','')
        if compressed:
            data=self.server.scenes[version][1]
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-cache')
        self.send_header('Content-Length',str(len(data)))
        if compressed:
            self.send_header('Content-Encoding','gzip')
            self.send_header('Vary','Accept-Encoding')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self,*args):
        pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--compare',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--tree-candidate',type=Path)
    parser.add_argument('--material-run',type=Path)
    args=parser.parse_args()
    httpd=ThreadingHTTPServer(('127.0.0.1',args.port),Viewer)
    httpd.scenes={}
    httpd.versions=[]
    httpd.material_scene=(args.material_run/'material_scene.json').read_bytes() if args.material_run else None
    calibration={}
    if args.calibration:
        for name in ('morphology','spatial_identity','occurrence'):
            calibration[name]=json.loads((args.calibration/(name+'_report.json')).read_text(encoding='utf-8'))
    httpd.calibration=json.dumps(calibration,ensure_ascii=False).encode('utf-8')
    for key,folder,label in [('current',args.run,'校准出生候选'),('tree',args.tree_candidate,'分枝规则候选 · 待验收'),('previous',args.compare,'初版空间候选')]:
        if folder is None: continue
        data=(folder/'scene.json').read_bytes()
        httpd.scenes[key]=(data,gzip.compress(data,compresslevel=3))
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        httpd.versions.append({'id':key,'label':label,'run':folder.name,
                              'spatial_calibrated':report.get('spatial_descriptors_calibrated',False)})
    print(f'CGB viewer ready: http://127.0.0.1:{args.port}',flush=True)
    httpd.serve_forever()

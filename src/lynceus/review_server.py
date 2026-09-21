"""A local static server for the reference review pages, with save-back.

A remote panel is annotated through an SSH tunnel, so the page has to be served rather than opened
as a file, and its save has to land on the machine that holds the panel rather than on whichever
laptop the browser runs on.

This is not part of the product. The annotator requires no server and this one never touches its
inference path; it exists so a human can build the reference the annotator is measured against. It
binds the loopback interface by default, because a panel directory holds imagery and there is no
reason to offer it to the network.
"""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAVE_NAME='reference.json'
MAX_BYTES=8*1024*1024


class ReviewHandler(SimpleHTTPRequestHandler):
    """Serves the panel and accepts a reference.json written back into one image's folder."""

    def __init__(self,*args,root,**kwargs):
        self.root=Path(root).resolve()
        super().__init__(*args,directory=str(self.root),**kwargs)

    def log_message(self,fmt,*args):
        if self.command!='GET':super().log_message(fmt,*args)

    def _target(self):
        """The reference file this request may write, or None.

        Only `<panel>/<existing image folder>/reference.json` is writable. The path is resolved and
        checked against the panel root, so a traversal cannot escape it, and the folder must already
        exist, so a request cannot invent one.
        """
        path=(self.root/self.path.lstrip('/')).resolve()
        if path.name!=SAVE_NAME:return None
        if not str(path).startswith(str(self.root)+'/'):return None
        if path.parent==self.root or not path.parent.is_dir():return None
        return path

    def do_PUT(self):
        target=self._target()
        if target is None:return self._fail(403,'only reference.json inside an image folder may be written')
        length=int(self.headers.get('Content-Length') or 0)
        if length<=0 or length>MAX_BYTES:return self._fail(413,'unsupported payload size')
        body=self.rfile.read(length)
        try: record=json.loads(body)
        except ValueError as exc: return self._fail(400,f'invalid json: {exc}')
        if not isinstance(record,dict):return self._fail(400,'not a reference record')
        # A save must be the record for this image, complete. Writing a partial one would strip the
        # identity the panel is checked on -- content hash, scene, near-duplicate group -- and the
        # loss would not surface until ingestion, long after the annotator moved on.
        missing=[k for k in ('id','sha256','scene_id','near_duplicate_group','size','visited_tiles','annotations')
                 if k not in record]
        if missing:return self._fail(400,'incomplete reference record, missing: '+','.join(missing))
        if record['id']!=target.parent.name:
            return self._fail(409,f"record is for {record['id']!r}, folder is {target.parent.name!r}")
        target.write_text(json.dumps(record,indent=1))
        payload=json.dumps({'saved':str(target.relative_to(self.root)),
                            'annotations':len(record['annotations']),
                            'visited_tiles':len(record['visited_tiles'])}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)

    do_POST=do_PUT

    def _fail(self,code,message):
        payload=message.encode()
        self.send_response(code);self.send_header('Content-Type','text/plain')
        self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)


def serve(root,host='127.0.0.1',port=8765):
    root=Path(root).resolve()
    if not (root/'index.html').is_file():raise ValueError('not_a_prepared_panel: '+str(root))
    server=ThreadingHTTPServer((host,port),partial(ReviewHandler,root=root))
    return server

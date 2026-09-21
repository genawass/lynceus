"""A local, offline review page for building an exhaustive reference panel.

This renders a self-contained HTML file. It loads no external resource -- no third-party script,
stylesheet or font, and no absolute URL of any kind -- which is what lets it run in the same
offline conditions the annotator itself requires.

It works two ways. Opened over `file://` it saves by download. Served by the local review server,
over an SSH tunnel for a remote panel, it saves with one same-origin request straight back into the
image's folder, so the file lands where ingestion looks for it instead of on whichever machine the
browser happens to run on. The request is relative and reaches only the process that served the
page; the page never addresses anything off the host.

It runs in one of two modes, and the mode decides what the finished panel can support.

Blind mode shows no prediction. The reference is annotated before the system's output is visible,
which is what makes it independent and what lets it measure recall.

Verification mode shows the system's retained boxes for the annotator to keep, fix or reject. It
buys precision, which a finite-category reference cannot give at all, and it costs recall: every
reference in the result is by construction something the system proposed, so nothing it says about
recall means anything. The page marks each box's origin -- accepted, adjusted or added -- so a box
that is still the model's geometry is distinguishable from one the annotator drew.

This is not the run report. `report.html` explains a finished run and is forbidden from requesting
annotations; this page exists outside the annotator's inference path, to build the reference that
the annotator is later measured against.

Completeness is driven by the tile sweep rather than by the annotator's sense of having finished:
the page walks the planned tiles, zooms each to fill the viewport so small entities are visible at
source resolution, and records which were reviewed. `reference.ingest` refuses a panel whose sweep
is incomplete, so the record has to be real.
"""
import json


def render_review_page(record, plan, ontology, image_href, predictions=None):
    """One self-contained review page for one image.

    `record` is the skeleton written by panel preparation, `plan` its review tiles, and
    `image_href` a path the browser can resolve relative to where the page is written.

    Passing `predictions` switches the page into verification mode. Each carries a box and a label,
    and nothing else from the run: no score, no uncertainty, no disposition. A number beside a box
    would steer the annotator toward the model's own confidence, which is the judgement being
    tested.
    """
    classes=[c for c in ontology['classes'] if c['id'] not in ('entity','stuff')]
    state={'record':{k:v for k,v in record.items() if k not in ('how_to_fill','planned_tiles')},
           'plan':plan,'classes':[{'id':c['id'],'name':c['name']} for c in classes],
           'excluded':ontology.get('excluded_by_convention',[]),'image':image_href,
           'mode':'verify' if predictions is not None else 'blind',
           'predictions':[{'id':p['id'],'bbox_xyxy':[float(v) for v in p['bbox_xyxy']],'label':p['label']}
                          for p in (predictions or [])]}
    return (_TEMPLATE
            .replace('__TITLE__',_escape(record['id']))
            .replace('__ONTOLOGY__',_escape(ontology['id']))
            .replace('__STATE__',json.dumps(state))
            ).encode()


def _escape(value):
    return (str(value).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))


_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Reference review - __TITLE__</title>
<style>
 :root{--bg:#16161a;--panel:#1e1e24;--line:#33333d;--text:#e8e8ee;--dim:#9a9aa8;--accent:#5aa9ff;--ok:#46c46a;--warn:#e8b339}
 *{box-sizing:border-box}
 body{margin:0;font:13px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;height:100vh;overflow:hidden}
 #side{width:290px;flex:none;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column}
 #side h1{font-size:14px;margin:12px 14px 2px}
 #side .sub{color:var(--dim);margin:0 14px 10px;font-size:11px}
 #tiles{overflow:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line);flex:1}
 .tile{padding:7px 14px;cursor:pointer;display:flex;justify-content:space-between;border-bottom:1px solid #26262e}
 .tile:hover{background:#26262e}
 .tile.current{background:#2b3a4d;box-shadow:inset 3px 0 0 var(--accent)}
 .tile .mark{color:var(--dim)}
 .tile.done .mark{color:var(--ok)}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 #bar{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 #stage{flex:1;overflow:hidden;position:relative;background:#0e0e11}
 canvas{position:absolute;top:0;left:0;cursor:crosshair}
 button{background:#2c2c36;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:5px 10px;cursor:pointer;font:inherit}
 button:hover{background:#383843}
 button.primary{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
 button.ghost{background:transparent}
 select,input{background:#2c2c36;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:4px 6px;font:inherit}
 #form{position:absolute;background:var(--panel);border:1px solid var(--accent);border-radius:8px;padding:10px;display:none;z-index:5;width:232px;box-shadow:0 8px 28px #0009}
 #form label{display:block;color:var(--dim);font-size:11px;margin:6px 0 2px}
 #form select{width:100%}
 #list{max-height:190px;overflow:auto;border-top:1px solid var(--line)}
 .row{padding:5px 14px;display:flex;justify-content:space-between;gap:6px;border-bottom:1px solid #26262e;font-size:12px}
 .row small{color:var(--dim)}
 .row button{padding:1px 6px;font-size:11px}
 #note{padding:10px 14px;color:var(--dim);font-size:11px;border-top:1px solid var(--line)}
 kbd{background:#2c2c36;border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:11px}
 .pill{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:99px;padding:1px 8px}
</style>
<div id="side">
  <h1>__TITLE__</h1>
  <p class="sub">ontology <b>__ONTOLOGY__</b><br><span id="progress"></span></p>
  <div id="tiles"></div>
  <div id="list"></div>
  <div id="verify" style="display:none;padding:10px 14px;border-top:1px solid var(--line)">
    <div style="color:var(--dim);font-size:11px;margin-bottom:6px">Proposal <b id="p-at"></b></div>
    <div style="display:flex;gap:6px;flex-wrap:wrap">
      <button id="v-keep" class="primary" style="flex:1">Keep <kbd>K</kbd></button>
      <button id="v-drop">Reject <kbd>X</kbd></button>
    </div>
    <div style="display:flex;gap:6px;margin-top:6px">
      <button id="v-edit" class="ghost" style="flex:1">Fix class / kind</button>
      <button id="v-redraw" class="ghost" style="flex:1">Redraw box</button>
    </div>
  </div>
  <div id="note">
    Drag on the image to draw a box. <kbd>Esc</kbd> cancels, <kbd>N</kbd> marks the tile reviewed and moves on.
    Mark an entity <b>unresolved</b> where the policy does not settle its extent or granularity — it stays
    in recall and leaves precision. Do not open the system's output until this file is finished.
  </div>
</div>
<div id="main">
  <div id="bar">
    <button id="prev" class="ghost">&larr;</button>
    <button id="next" class="primary">Reviewed, next &rarr;</button>
    <button id="whole" class="ghost">Whole image</button>
    <span class="pill" id="where"></span>
    <span style="flex:1"></span>
    <button id="load" class="ghost">Load</button>
    <button id="save" class="primary">Save reference.json</button>
    <span class="pill" id="saved"></span>
    <input id="file" type="file" accept="application/json" style="display:none">
  </div>
  <div id="stage"><canvas id="cv"></canvas></div>
</div>
<div id="form">
  <label>Class</label><select id="f-label"></select>
  <label>Kind</label><select id="f-kind">
    <option value="instance">instance — a whole entity</option>
    <option value="part">part — belongs to another entity</option>
    <option value="group">group — inseparable, count unresolved</option>
    <option value="stuff">stuff — continuous material</option></select>
  <label>Scene layer</label><select id="f-layer">
    <option value="physical">physical</option>
    <option value="reflected">reflected</option>
    <option value="depicted">depicted</option></select>
  <label>Resolution</label><select id="f-res">
    <option value="resolved">resolved</option>
    <option value="unresolved">unresolved — policy does not settle it</option></select>
  <div style="display:flex;gap:6px;margin-top:10px">
    <button id="f-add" class="primary" style="flex:1">Add</button>
    <button id="f-cancel" class="ghost">Cancel</button>
  </div>
</div>
<script>
const S = __STATE__;
const rec = S.record, plan = S.plan;
rec.visited_tiles = rec.visited_tiles || [];
rec.annotations = rec.annotations || [];
const visited = new Set(rec.visited_tiles);
let current = 0, view = null, drag = null, pending = null, seq = 0;
const MODE = S.mode;
const proposals = (S.predictions || []).map(p => ({...p, verdict: null}));
let at = 0, redrawing = null;
rec.rejected = rec.rejected || [];

const img = new Image();
img.src = S.image;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const stage = document.getElementById('stage');

for (const c of S.classes) {
  const o = document.createElement('option');
  o.value = c.id; o.textContent = c.name; document.getElementById('f-label').appendChild(o);
}

function viewport(){ return {w: stage.clientWidth, h: stage.clientHeight}; }

function setView(box){
  const vp = viewport();
  const scale = Math.min(vp.w / (box[2]-box[0]), vp.h / (box[3]-box[1]));
  view = {x: box[0], y: box[1], scale: scale};
  cv.width = vp.w; cv.height = vp.h;
  draw();
}

function toImage(px, py){ return [view.x + px/view.scale, view.y + py/view.scale]; }
function toScreen(ix, iy){ return [(ix-view.x)*view.scale, (iy-view.y)*view.scale]; }

function draw(){
  if (!view) return;
  ctx.fillStyle = '#0e0e11'; ctx.fillRect(0,0,cv.width,cv.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, view.x, view.y, cv.width/view.scale, cv.height/view.scale,
                0, 0, cv.width, cv.height);
  // the tile under review, so its edges are unambiguous
  const t = plan[current].bbox_xyxy;
  const a = toScreen(t[0],t[1]), b = toScreen(t[2],t[3]);
  ctx.strokeStyle = '#5aa9ff88'; ctx.lineWidth = 2;
  ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);
  if (MODE === 'verify'){
    proposals.forEach((p, i) => {
      if (p.verdict) return;                      // decided ones leave the overlay
      const a = toScreen(p.bbox_xyxy[0], p.bbox_xyxy[1]), b = toScreen(p.bbox_xyxy[2], p.bbox_xyxy[3]);
      ctx.strokeStyle = i === at ? '#ffffff' : '#7a7a8a';
      ctx.lineWidth = i === at ? 2 : 1;
      if (i !== at) ctx.setLineDash([3,3]);
      ctx.strokeRect(a[0], a[1], b[0]-a[0], b[1]-a[1]);
      ctx.setLineDash([]);
      if (i === at){ ctx.fillStyle = '#ffffff'; ctx.font = '11px system-ui'; ctx.fillText(p.label, a[0]+2, a[1]-3); }
    });
  }
  for (const e of rec.annotations){
    const p = toScreen(e.bbox_xyxy[0],e.bbox_xyxy[1]), q = toScreen(e.bbox_xyxy[2],e.bbox_xyxy[3]);
    ctx.strokeStyle = e.resolution === 'unresolved' ? '#e8b339'
                    : e.kind !== 'instance' || e.scene_layer !== 'physical' ? '#9a9aa8' : '#46c46a';
    ctx.lineWidth = 2; ctx.strokeRect(p[0],p[1],q[0]-p[0],q[1]-p[1]);
    ctx.fillStyle = ctx.strokeStyle; ctx.font = '11px system-ui';
    ctx.fillText(e.label + (e.kind!=='instance' ? ' ['+e.kind+']' : ''), p[0]+2, p[1]-3);
  }
  if (drag){
    ctx.strokeStyle = '#ffffff'; ctx.setLineDash([4,3]); ctx.lineWidth = 1;
    ctx.strokeRect(drag.x0, drag.y0, drag.x1-drag.x0, drag.y1-drag.y0); ctx.setLineDash([]);
  }
}

function renderSide(){
  const tl = document.getElementById('tiles'); tl.innerHTML = '';
  plan.forEach((t,i) => {
    const d = document.createElement('div');
    d.className = 'tile' + (i===current?' current':'') + (visited.has(t.id)?' done':'');
    d.innerHTML = '<span>'+t.id+'</span><span class="mark">'+(visited.has(t.id)?'reviewed':'&middot;')+'</span>';
    d.onclick = () => { current = i; setView(t.bbox_xyxy); renderSide(); };
    tl.appendChild(d);
  });
  const decided = proposals.filter(p => p.verdict).length;
  document.getElementById('progress').innerHTML =
    (MODE === 'verify'
      ? decided + ' of ' + proposals.length + ' proposals judged &middot; ' + rec.rejected.length + ' rejected<br>'
      : '') +
    visited.size + ' of ' + plan.length + ' tiles reviewed &middot; ' + rec.annotations.length + ' entities';
  document.getElementById('where').textContent = plan[current].id;
  const ls = document.getElementById('list'); ls.innerHTML = '';
  rec.annotations.slice().reverse().forEach(e => {
    const r = document.createElement('div'); r.className = 'row';
    r.innerHTML = '<span>'+e.label+' <small>'+e.kind+'/'+e.resolution+'</small></span>';
    const b = document.createElement('button'); b.textContent = 'remove';
    b.onclick = () => { rec.annotations = rec.annotations.filter(x => x.id !== e.id); renderSide(); draw(); };
    r.appendChild(b); ls.appendChild(r);
  });
}

cv.addEventListener('mousedown', ev => {
  const r = cv.getBoundingClientRect();
  drag = {x0: ev.clientX-r.left, y0: ev.clientY-r.top, x1: ev.clientX-r.left, y1: ev.clientY-r.top};
});
cv.addEventListener('mousemove', ev => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  drag.x1 = ev.clientX-r.left; drag.y1 = ev.clientY-r.top; draw();
});
cv.addEventListener('mouseup', () => {
  if (!drag) return;
  const p = toImage(Math.min(drag.x0,drag.x1), Math.min(drag.y0,drag.y1));
  const q = toImage(Math.max(drag.x0,drag.x1), Math.max(drag.y0,drag.y1));
  const box = [p[0], p[1], q[0], q[1]];
  drag = null;
  if (box[2]-box[0] < 1 || box[3]-box[1] < 1){ draw(); return; }
  pending = box.map(v => Math.max(0, Math.round(v*100)/100));
  pending[2] = Math.min(pending[2], rec.size[0]); pending[3] = Math.min(pending[3], rec.size[1]);
  const f = document.getElementById('form');
  const anchor = toScreen(pending[2], pending[1]);
  f.style.display = 'block';
  f.style.left = Math.max(4, Math.min(stage.clientWidth-240, anchor[0]+306))+'px';
  f.style.top = Math.max(50, Math.min(stage.clientHeight-250, anchor[1]+50))+'px';
  document.getElementById('f-label').focus();
  draw();
});

document.getElementById('f-add').onclick = () => {
  if (!pending) return;
  const entry = {id: 'ref-' + (seq++) + '-' + Date.now().toString(36),
    bbox_xyxy: pending,
    label: document.getElementById('f-label').value,
    kind: document.getElementById('f-kind').value,
    scene_layer: document.getElementById('f-layer').value,
    resolution: document.getElementById('f-res').value,
    origin: redrawing || 'added'};
  pending = null; redrawing = null;
  document.getElementById('form').style.display = 'none';
  if (MODE === 'verify' && at >= 0 && proposals[at] && !proposals[at].verdict && entry.origin !== 'added'){
    decide('keep', entry);
  } else {
    rec.annotations.push(entry); renderSide(); draw();
  }
};
document.getElementById('f-cancel').onclick = () => {
  pending = null; document.getElementById('form').style.display = 'none'; draw();
};

function focusProposal(){
  if (MODE !== 'verify') return;
  const undecided = proposals.findIndex((p,i) => i >= at && !p.verdict);
  at = undecided >= 0 ? undecided : proposals.findIndex(p => !p.verdict);
  document.getElementById('verify').style.display = at >= 0 ? 'block' : 'none';
  if (at < 0){ renderSide(); draw(); return; }
  const p = proposals[at];
  // Centre the view on the proposal under judgement, keeping the tile's magnification.
  const pad = 80;
  setView([Math.max(0, p.bbox_xyxy[0]-pad), Math.max(0, p.bbox_xyxy[1]-pad),
           Math.min(rec.size[0], p.bbox_xyxy[2]+pad), Math.min(rec.size[1], p.bbox_xyxy[3]+pad)]);
  document.getElementById('p-at').textContent =
    (proposals.filter(x => x.verdict).length + 1) + ' of ' + proposals.length;
  renderSide();
}

function decide(verdict, entry){
  const p = proposals[at];
  p.verdict = verdict;
  if (verdict === 'keep') rec.annotations.push(entry);
  else rec.rejected.push({id: p.id, bbox_xyxy: p.bbox_xyxy, label: p.label});
  focusProposal(); draw();
}

function keepAsIs(){
  const p = proposals[at];
  decide('keep', {id: 'ref-' + p.id, bbox_xyxy: p.bbox_xyxy, label: p.label,
                  kind: 'instance', scene_layer: 'physical', resolution: 'resolved',
                  // Still the model's geometry: localization cannot be scored against it.
                  origin: 'accepted'});
}

function markAndNext(){
  visited.add(plan[current].id);
  rec.visited_tiles = plan.map(t => t.id).filter(id => visited.has(id));
  if (current < plan.length-1) current++;
  setView(plan[current].bbox_xyxy); renderSide();
}
document.getElementById('next').onclick = markAndNext;
if (MODE === 'verify'){
  document.getElementById('v-keep').onclick = keepAsIs;
  document.getElementById('v-drop').onclick = () => decide('drop', null);
  document.getElementById('v-edit').onclick = () => {
    const p = proposals[at];
    pending = p.bbox_xyxy; redrawing = 'accepted';
    document.getElementById('f-label').value = p.label;
    const f = document.getElementById('form');
    f.style.display = 'block'; f.style.left = '320px'; f.style.top = '60px';
  };
  document.getElementById('v-redraw').onclick = () => {
    redrawing = 'adjusted';
    document.getElementById('note').scrollIntoView();
  };
}
document.getElementById('prev').onclick = () => {
  if (current > 0) current--;
  setView(plan[current].bbox_xyxy); renderSide();
};
document.getElementById('whole').onclick = () => setView([0,0,rec.size[0],rec.size[1]]);

document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape'){ pending = null; drag = null; document.getElementById('form').style.display='none'; draw(); }
  const typing = document.getElementById('form').style.display === 'block';
  if (ev.key.toLowerCase() === 'n' && !typing) markAndNext();
  if (MODE === 'verify' && !typing && at >= 0){
    if (ev.key.toLowerCase() === 'k') keepAsIs();
    if (ev.key.toLowerCase() === 'x') decide('drop', null);
  }
});

function download(text){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type:'application/json'}));
  a.download = 'reference.json'; a.click();
}
document.getElementById('save').onclick = async () => {
  rec.visited_tiles = plan.map(t => t.id).filter(id => visited.has(id));
  const text = JSON.stringify(rec, null, 1);
  const status = document.getElementById('saved');
  // Served over the local review server, write back beside the image so the file lands where
  // ingestion looks. Opened as a plain file there is nothing to write to, so fall back to a
  // download and say which happened rather than appearing to have saved.
  if (location.protocol !== 'file:'){
    try {
      const r = await fetch('reference.json', {method:'PUT', headers:{'Content-Type':'application/json'}, body:text});
      if (!r.ok) throw new Error(await r.text());
      status.textContent = 'saved to panel ' + new Date().toLocaleTimeString();
      status.style.color = '#46c46a';
      return;
    } catch (e) {
      status.textContent = 'server save failed, downloaded instead: ' + e.message;
      status.style.color = '#e8b339';
    }
  } else {
    status.textContent = 'downloaded; copy it into this image\'s folder';
    status.style.color = '#9a9aa8';
  }
  download(text);
};
document.getElementById('load').onclick = () => document.getElementById('file').click();
document.getElementById('file').onchange = ev => {
  const f = ev.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    const d = JSON.parse(r.result);
    rec.annotations = d.annotations || [];
    visited.clear(); (d.visited_tiles||[]).forEach(t => visited.add(t));
    renderSide(); draw();
  };
  r.readAsText(f);
};

img.onload = () => {
  setView(plan[0].bbox_xyxy); renderSide();
  if (MODE === 'verify') focusProposal();
};
window.addEventListener('resize', () => setView(plan[current].bbox_xyxy));
</script>
"""

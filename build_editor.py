"""Rebuild the video editor UI in index.html"""
path = r'C:\Users\ashet\OneDrive\Desktop\ai-video-platform\frontend\index.html'
with open(path, encoding='utf-8') as f:
    text = f.read()

# ── NEW EDITOR CSS ─────────────────────────────────────────────────────────────
editor_css = """
/* ═══ PROFESSIONAL VIDEO EDITOR ══════════════════════════════════ */
#view-generate.view.active{display:flex;flex-direction:column;overflow:hidden;}
#view-generate{height:calc(100vh - 52px - 48px);overflow:hidden;}
.ed-empty{flex:1;display:flex;align-items:center;justify-content:center;}
.ed-dropzone{width:500px;max-width:90%;padding:64px 40px;display:flex;flex-direction:column;align-items:center;gap:14px;border:2px dashed var(--b2);border-radius:22px;cursor:pointer;position:relative;background:var(--bg1);transition:all .2s;text-align:center;}
.ed-dropzone:hover,.ed-dropzone.drag{border-color:rgba(91,95,217,.5);background:var(--vg);}
.ed-dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.ed-drop-icon{font-size:54px;opacity:.12;line-height:1;}
.ed-drop-title{font-size:22px;font-weight:800;color:var(--t);letter-spacing:-.4px;}
.ed-drop-sub{font-size:14px;color:var(--t3);}
.ed-drop-fmts{display:flex;gap:8px;}
.ed-drop-fmt{padding:3px 10px;border:1px solid var(--b2);border-radius:5px;font-size:11px;font-weight:700;color:var(--t3);font-family:monospace;}
.ed-app{display:flex;flex-direction:column;flex:1;overflow:hidden;}
.ed-toolbar{display:flex;align-items:center;gap:6px;padding:8px 14px;background:var(--bg1);border-bottom:1px solid var(--b);flex-shrink:0;height:52px;}
.ed-tool-btn{display:flex;align-items:center;gap:7px;padding:7px 16px;border-radius:8px;border:1.5px solid var(--b2);background:var(--bg2);font-size:12px;font-weight:700;color:var(--t2);cursor:pointer;font-family:inherit;transition:all .15s;}
.ed-tool-btn.on{background:var(--v);border-color:var(--v);color:#fff;box-shadow:0 4px 12px rgba(91,95,217,.3);}
.ed-tool-btn:hover:not(.on){border-color:var(--v);color:var(--v);background:var(--vg);}
.ed-tb-div{width:1px;height:26px;background:var(--b2);margin:0 4px;}
.ed-new-btn{padding:6px 12px;border-radius:7px;border:1px solid var(--b2);background:transparent;font-size:11px;font-weight:600;color:var(--t3);cursor:pointer;font-family:inherit;transition:all .15s;}
.ed-new-btn:hover{color:var(--t2);}
.ed-spacer{flex:1;}
.ed-dl-btn{display:none;padding:7px 16px;border-radius:8px;background:var(--g);color:#fff;border:none;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit;align-items:center;gap:6px;}
.ed-dl-btn.show{display:flex;}
.ed-proc-btn{padding:7px 20px;border-radius:8px;background:linear-gradient(135deg,var(--v),var(--c));color:#fff;border:none;font-size:13px;font-weight:800;cursor:pointer;font-family:inherit;transition:all .2s;position:relative;overflow:hidden;}
.ed-proc-btn:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(91,95,217,.35);}
.ed-proc-btn:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none;}
.ed-proc-btn.running::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent);animation:sw 1.5s infinite;}
.ed-main{display:flex;flex:1;min-height:0;}
.ed-preview{flex:1;background:#08080f;position:relative;overflow:hidden;cursor:crosshair;display:flex;align-items:center;justify-content:center;min-height:0;}
.ed-preview video{max-width:100%;max-height:100%;display:block;}
#objCanvas{position:absolute;top:0;left:0;pointer-events:none;}
.ed-play-ov{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:62px;height:62px;border-radius:50%;background:rgba(255,255,255,.12);backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;font-size:20px;color:#fff;cursor:pointer;opacity:0;transition:opacity .2s;}
.ed-preview:hover .ed-play-ov{opacity:1;}
.ed-vid-badge{position:absolute;bottom:10px;left:10px;background:rgba(0,0,0,.65);color:rgba(255,255,255,.8);font-size:11px;padding:4px 10px;border-radius:6px;font-family:monospace;}
.ed-panel{width:276px;background:var(--bg1);border-left:1px solid var(--b);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0;}
.ep-sec{padding:14px 16px;border-bottom:1px solid var(--b);}
.ep-lbl{font-size:10px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;}
.ep-hint-box{display:flex;flex-direction:column;align-items:center;text-align:center;gap:10px;padding:28px 16px;}
.ep-hint-icon{font-size:38px;opacity:.18;}
.ep-hint-txt{font-size:12px;color:var(--t3);line-height:1.8;}
.ep-chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.ep-rescan{width:100%;padding:6px;border:1px solid var(--b2);border-radius:7px;background:transparent;color:var(--t3);font-size:11px;font-weight:700;cursor:pointer;font-family:inherit;transition:all .15s;}
.ep-rescan:hover{color:var(--v);border-color:var(--v);background:var(--vg);}
.ep-obj-header{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.ep-obj-icon{width:40px;height:40px;border-radius:10px;background:var(--vg);display:flex;align-items:center;justify-content:center;font-size:18px;border:1px solid rgba(91,95,217,.2);flex-shrink:0;}
.ep-obj-name{font-size:14px;font-weight:800;color:var(--t);}
.ep-obj-conf{font-size:11px;color:var(--g);font-family:monospace;}
.ep-clear-btn{background:none;border:none;color:var(--t3);font-size:20px;cursor:pointer;padding:0 4px;margin-left:auto;line-height:1;}
.ep-pbar-wrap{height:4px;background:var(--bg3);border-radius:2px;overflow:hidden;margin:6px 0;}
.ep-pbar{height:100%;background:linear-gradient(90deg,var(--v),var(--c));width:0%;transition:width .4s;border-radius:2px;}
.ep-pnote{font-size:11px;color:var(--t3);line-height:1.65;}
.ep-done-ok{display:flex;align-items:center;gap:6px;color:var(--g);font-weight:700;font-size:14px;margin-bottom:10px;}
.ep-foot{padding:12px 16px;border-top:1px solid var(--b);flex-shrink:0;}
.ed-tl-area{background:var(--bg1);border-top:2px solid var(--b);padding:10px 16px 16px;flex-shrink:0;}
.ed-ctl-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.tl-timecode{font-size:12px;color:var(--t2);font-family:monospace;font-weight:600;}
.ed-tl-wrap{position:relative;padding-bottom:22px;user-select:none;}
.tl-track-bar{height:40px;background:#dde5fb;border-radius:8px;position:relative;overflow:visible;border:1px solid rgba(91,95,217,.25);cursor:crosshair;}
.tl-gray{position:absolute;top:0;height:100%;background:rgba(30,41,82,.18);pointer-events:none;z-index:1;border-radius:8px;}
.tl-gray-l{left:0;}
.tl-gray-r{right:0;}
.tl-remove-zone{position:absolute;top:4px;height:calc(100% - 8px);background:rgba(220,38,38,.22);border:1.5px solid rgba(220,38,38,.6);border-radius:5px;pointer-events:none;z-index:2;}
.tl-trim-h{position:absolute;top:-6px;bottom:-6px;width:12px;background:var(--v);border-radius:6px;z-index:10;cursor:ew-resize;border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.25);}
.tl-trim-h.left{transform:translateX(-50%);}
.tl-trim-h.right{transform:translateX(50%);}
.tl-edit-h{position:absolute;top:2px;height:calc(100% - 4px);width:8px;background:rgba(220,38,38,.85);border-radius:4px;z-index:11;cursor:ew-resize;display:none;}
.tl-edit-h.left{transform:translateX(-50%);}
.tl-edit-h.right{transform:translateX(50%);}
.tl-ph{position:absolute;top:-6px;bottom:-6px;width:3px;background:#1e2a4a;border-radius:2px;z-index:20;transform:translateX(-50%);pointer-events:none;}
.tl-ph::before{content:'';position:absolute;top:6px;left:50%;transform:translateX(-50%);width:12px;height:12px;background:#1e2a4a;border-radius:50%;}
.tl-ruler-row{display:flex;justify-content:space-between;padding:3px 0 0;}
.tl-tick{font-size:10px;color:var(--t3);font-family:monospace;}
"""

text = text.replace('\n</style>', editor_css + '\n</style>', 1)
print("CSS added")

# ── NEW GENERATE VIEW HTML ──────────────────────────────────────────────────────
new_html = """        <!-- GENERATE - Video Editor -->
        <div id="view-generate" class="view">
          <div class="ed-empty" id="edEmpty">
            <div class="ed-dropzone" id="editorDrop"
                 ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)">
              <input type="file" accept="video/*" id="videoInput" onchange="handleFileSelect(event)">
              <div class="ed-drop-icon">&#9654;</div>
              <div class="ed-drop-title">Drop your video here</div>
              <div class="ed-drop-sub">or click to browse &mdash; click any object to remove it</div>
              <div class="ed-drop-fmts">
                <span class="ed-drop-fmt">MP4</span><span class="ed-drop-fmt">MOV</span>
                <span class="ed-drop-fmt">AVI</span><span class="ed-drop-fmt">MKV</span>
              </div>
            </div>
          </div>
          <div class="ed-app" id="edApp" style="display:none;">
            <div class="ed-toolbar">
              <button class="ed-tool-btn on" id="edBtnRemove" onclick="setTool('remove')">&#128465; Remove Object</button>
              <button class="ed-tool-btn" id="edBtnTrim" onclick="setTool('trim')">&#9986; Trim Video</button>
              <div class="ed-tb-div"></div>
              <button class="ed-new-btn" onclick="resetEditor()">&#43; New Video</button>
              <div class="ed-spacer"></div>
              <button class="ed-dl-btn" id="edDlBtn" onclick="downloadResult()">&#11015; Download</button>
              <button class="ed-proc-btn" id="edProcBtn" onclick="edProcess()">&#9889; Process</button>
            </div>
            <div class="ed-main">
              <div class="ed-preview" id="edPreview" onclick="handleStageClick(event)">
                <video id="mainVideo" preload="metadata"></video>
                <canvas id="objCanvas"></canvas>
                <div id="objPopup" style="display:none;position:absolute;z-index:30;" onclick="event.stopPropagation()">
                  <div class="obj-popup-inner">
                    <div class="popup-tag">
                      <span id="popupName">Object</span>
                      <span class="popup-tag-conf" id="popupConf">91%</span>
                    </div>
                    <div class="popup-actions">
                      <button class="popup-btn on" onclick="pickAndClose('remove')">&#128465; Remove</button>
                      <button class="popup-btn" onclick="pickAndClose('blur')">&#127807; Blur</button>
                    </div>
                  </div>
                </div>
                <div class="ed-play-ov" onclick="togglePlay()">&#9654;</div>
                <div class="ed-vid-badge" id="edVidBadge"></div>
              </div>
              <div class="ed-panel">
                <!-- REMOVE PANEL -->
                <div id="panelRemove" style="display:flex;flex-direction:column;flex:1;overflow-y:auto;">
                  <div class="ep-sec">
                    <div class="ep-lbl">Detected Objects</div>
                    <div class="ep-chips" id="objChips"></div>
                    <button class="ep-rescan" onclick="runDetection()">&#8635; Rescan</button>
                  </div>
                  <div class="ep-sec" id="epSelSec" style="display:none;">
                    <div class="ep-obj-header">
                      <div class="ep-obj-icon" id="selIcon">?</div>
                      <div style="flex:1;min-width:0;">
                        <div class="ep-obj-name" id="selName">Object</div>
                        <div class="ep-obj-conf" id="selConf"></div>
                      </div>
                      <button class="ep-clear-btn" onclick="clearSelection()">&#215;</button>
                    </div>
                    <div class="ep-lbl">Action</div>
                    <div class="action-grid">
                      <div class="action-card on" id="ac-remove" onclick="setEditorAction('remove')">
                        <div class="action-card-icon">&#128465;</div>
                        <div class="action-card-label">Remove</div>
                        <div class="action-card-sub">AI fills background</div>
                      </div>
                      <div class="action-card" id="ac-blur" onclick="setEditorAction('blur')">
                        <div class="action-card-icon">&#127807;</div>
                        <div class="action-card-label">Blur</div>
                        <div class="action-card-sub">Censor it</div>
                      </div>
                    </div>
                    <div class="ep-lbl" style="margin-top:12px;">Object disappears from &#8594; to</div>
                    <div class="time-range-row">
                      <div class="tr-field"><label>FROM (s)</label>
                        <input type="number" id="startTime" value="0" min="0" step="0.5" oninput="syncRangeFromInputs()">
                      </div>
                      <div class="tr-sep">&#8594;</div>
                      <div class="tr-field"><label>TO (s)</label>
                        <input type="number" id="endTime" value="10" min="0" step="0.5" oninput="syncRangeFromInputs()">
                      </div>
                    </div>
                    <div class="tr-quick">
                      <button onclick="setRangeFromNow(5)">+5s</button>
                      <button onclick="setRangeFromNow(10)">+10s</button>
                      <button onclick="setFullRange()">Full video</button>
                    </div>
                  </div>
                  <div class="ep-sec" id="epProgress" style="display:none;">
                    <div class="ep-lbl" id="epProgLabel">Processing...</div>
                    <div class="ep-pbar-wrap"><div class="ep-pbar" id="epProgBar"></div></div>
                    <div class="ep-pnote">Full original video is output with the object removed.</div>
                  </div>
                  <div class="ep-sec" id="epResult" style="display:none;">
                    <div class="ep-done-ok">&#10003; Object removed!</div>
                    <button class="dl-btn" onclick="downloadResult()" style="margin-bottom:6px;">&#11015; Download Full Video</button>
                    <button class="reset-btn" onclick="clearResult()">New Edit</button>
                  </div>
                  <div class="ep-hint-box" id="epHint">
                    <div class="ep-hint-icon">&#128247;</div>
                    <div class="ep-hint-txt">Click any object in the video to select it, then set when it disappears and hit Process.</div>
                  </div>
                  <div class="ep-foot">
                    <button class="ep-proc" id="procBtn" onclick="edProcess()">&#9889; Remove Object from Video</button>
                  </div>
                </div>
                <!-- TRIM PANEL -->
                <div id="panelTrim" style="display:none;flex-direction:column;flex:1;">
                  <div class="ep-sec">
                    <div class="ep-lbl">Trim Video</div>
                    <div class="ep-hint-txt" style="margin-bottom:12px;">Drag the white handles on the timeline to set your trim points.</div>
                    <div class="ep-lbl">Keep from &#8594; to</div>
                    <div class="time-range-row">
                      <div class="tr-field"><label>IN (s)</label>
                        <input type="number" id="trimStart" value="0" min="0" step="0.5" oninput="syncTrimFromInputs()">
                      </div>
                      <div class="tr-sep">&#8594;</div>
                      <div class="tr-field"><label>OUT (s)</label>
                        <input type="number" id="trimEnd" value="10" min="0" step="0.5" oninput="syncTrimFromInputs()">
                      </div>
                    </div>
                    <div style="margin-top:8px;font-size:12px;color:var(--t3);" id="trimDurLbl">Duration: 10s</div>
                  </div>
                  <div class="ep-sec" id="trimProgress" style="display:none;">
                    <div class="ep-lbl">Trimming...</div>
                    <div class="ep-pbar-wrap"><div class="ep-pbar" style="width:100%;animation:sw 1.5s infinite;"></div></div>
                  </div>
                  <div class="ep-sec" id="trimResult" style="display:none;">
                    <div class="ep-done-ok">&#10003; Trim complete!</div>
                    <button class="dl-btn" onclick="downloadResult()">&#11015; Download Trimmed Video</button>
                  </div>
                  <div class="ep-foot" style="margin-top:auto;">
                    <button class="ep-proc" id="trimProcBtn" onclick="processTrim()">&#9986; Export Trimmed Video</button>
                  </div>
                </div>
              </div>
            </div>
            <!-- TIMELINE -->
            <div class="ed-tl-area">
              <div class="ed-ctl-row">
                <button class="play-btn" id="playBtn" onclick="togglePlay()">&#9654;</button>
                <span class="tl-timecode" id="tlTimecode">0:00 / 0:00</span>
              </div>
              <div class="ed-tl-wrap" id="edTlWrap"
                   onmousedown="tlDown(event)" onmousemove="tlMove(event)"
                   onmouseup="tlUp()" onmouseleave="tlUp()">
                <div class="tl-track-bar" id="tlTrack">
                  <div class="tl-gray tl-gray-l" id="tlGrayL" style="width:0%;"></div>
                  <div class="tl-gray tl-gray-r" id="tlGrayR" style="width:0%;"></div>
                  <div class="tl-remove-zone" id="tlRemoveZone" style="display:none;"></div>
                  <div class="tl-trim-h left" id="tlTrimL" style="left:0%;"></div>
                  <div class="tl-trim-h right" id="tlTrimR" style="left:100%;"></div>
                  <div class="tl-edit-h left" id="tlEditL"></div>
                  <div class="tl-edit-h right" id="tlEditR"></div>
                  <div class="tl-ph" id="tlPh" style="left:0%;"></div>
                </div>
                <div class="tl-ruler-row" id="tlRuler"></div>
              </div>
            </div>
          </div>
        </div>"""

s = text.find('<!-- GENERATE VIEW')
if s == -1:
    s = text.find('id="view-generate"')
    s = text.rfind('<div', 0, s) - 8
s = max(0, s)
e = text.find('id="view-history"')
e = text.rfind('        <div', 0, e)
print(f"HTML replace: {s}..{e}")
text = text[:s] + new_html.strip() + '\n\n        ' + text[e:]

# ── NEW GENERATE JS ─────────────────────────────────────────────────────────────
new_js = """
/* VIDEO EDITOR JS ================================================= */
let _rafId=null,_currentTool='remove';
let _trimS=0,_trimE=1,_editS=0,_editE=0,_dragging=null;

function handleDragOver(e){e.preventDefault();document.getElementById('editorDrop').classList.add('drag');}
function handleDragLeave(){document.getElementById('editorDrop').classList.remove('drag');}
function handleDrop(e){e.preventDefault();document.getElementById('editorDrop').classList.remove('drag');const f=e.dataTransfer.files[0];if(f&&f.type.startsWith('video/'))uploadFile(f);}
function handleFileSelect(e){if(e.target.files[0])uploadFile(e.target.files[0]);}

async function uploadFile(file){
  currentFile=file;
  document.getElementById('edEmpty').style.display='none';
  document.getElementById('edApp').style.display='flex';
  try{
    const fd=new FormData();fd.append('file',file);
    const r=await fetch(API+'/upload-video',{method:'POST',body:fd});
    const d=await r.json();
    if(d.success){showVideo(file);showToast('Uploaded — scanning for objects...','success');setTimeout(()=>runDetection(),600);}
  }catch(err){
    showToast('Cannot connect to server','error');
    document.getElementById('edEmpty').style.display='flex';
    document.getElementById('edApp').style.display='none';
  }
}

function showVideo(file){
  const v=document.getElementById('mainVideo');
  v.src=URL.createObjectURL(file);
  v.onloadedmetadata=()=>{
    videoDuration=v.duration;
    _trimS=0;_trimE=1;_editS=0;_editE=Math.min(1,10/v.duration);
    document.getElementById('trimStart').value='0';
    document.getElementById('trimEnd').value=Math.floor(v.duration);
    document.getElementById('startTime').value='0';
    document.getElementById('endTime').value=Math.min(10,Math.floor(v.duration));
    document.getElementById('edVidBadge').textContent=v.videoWidth+'x'+v.videoHeight+' · '+fmtTime(v.duration);
    buildRuler();startCanvasLoop();updateTimeline();
  };
  v.addEventListener('timeupdate',()=>{
    if(!v.duration)return;
    document.getElementById('tlPh').style.left=(v.currentTime/v.duration*100)+'%';
    document.getElementById('tlTimecode').textContent=fmtTime(v.currentTime)+' / '+fmtTime(v.duration);
  });
  v.addEventListener('ended',()=>{document.getElementById('playBtn').innerHTML='&#9654;';});
}

function startCanvasLoop(){if(_rafId)cancelAnimationFrame(_rafId);function loop(){drawBboxes();_rafId=requestAnimationFrame(loop);}_rafId=requestAnimationFrame(loop);}
function stopCanvasLoop(){if(_rafId){cancelAnimationFrame(_rafId);_rafId=null;}}

function drawBboxes(){
  const canvas=document.getElementById('objCanvas');const video=document.getElementById('mainVideo');
  if(!canvas||!video||!video.videoWidth)return;
  const vRect=video.getBoundingClientRect();const sRect=document.getElementById('edPreview').getBoundingClientRect();
  const offL=vRect.left-sRect.left,offT=vRect.top-sRect.top;
  const dw=Math.round(vRect.width),dh=Math.round(vRect.height);
  canvas.style.left=offL+'px';canvas.style.top=offT+'px';canvas.style.width=dw+'px';canvas.style.height=dh+'px';
  if(canvas.width!==dw||canvas.height!==dh){canvas.width=dw;canvas.height=dh;}
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,dw,dh);
  if(!detectedObjects.length)return;
  const sx=dw/video.videoWidth,sy=dh/video.videoHeight;
  detectedObjects.forEach(obj=>{
    const[x1,y1,x2,y2]=obj.bbox;const dx=x1*sx,dy=y1*sy,bw=(x2-x1)*sx,bh=(y2-y1)*sy;const sel=selectedObject===obj;
    ctx.save();ctx.fillStyle=sel?'rgba(91,95,217,.2)':'rgba(91,95,217,.07)';ctx.strokeStyle=sel?'#5b5fd9':'rgba(91,95,217,.6)';ctx.lineWidth=sel?3:1.5;
    ctx.fillRect(dx,dy,bw,bh);ctx.strokeRect(dx,dy,bw,bh);
    const lbl=obj.label+' '+Math.round(obj.confidence*100)+'%';ctx.font='bold '+(sel?13:11)+'px sans-serif';
    const tw=ctx.measureText(lbl).width;const ly=Math.max(18,dy-2);
    ctx.fillStyle=sel?'#5b5fd9':'rgba(91,95,217,.85)';ctx.fillRect(dx-1,ly-14,tw+10,18);ctx.fillStyle='#fff';ctx.fillText(lbl,dx+4,ly);
    if(sel){[[dx,dy],[dx+bw,dy],[dx,dy+bh],[dx+bw,dy+bh]].forEach(([cx,cy])=>{ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);ctx.fillStyle='#5b5fd9';ctx.fill();});}
    ctx.restore();
  });
}

function handleStageClick(e){
  const video=document.getElementById('mainVideo');
  if(!video.videoWidth||!detectedObjects.length)return;
  const vRect=video.getBoundingClientRect();const cx=e.clientX-vRect.left,cy=e.clientY-vRect.top;
  if(cx<0||cy<0||cx>vRect.width||cy>vRect.height){hideObjPopup();return;}
  const ox=cx*(video.videoWidth/vRect.width),oy=cy*(video.videoHeight/vRect.height);
  let hit=null,minArea=Infinity;
  for(const obj of detectedObjects){const[x1,y1,x2,y2]=obj.bbox;if(ox>=x1&&ox<=x2&&oy>=y1&&oy<=y2){const a=(x2-x1)*(y2-y1);if(a<minArea){minArea=a;hit=obj;}}}
  if(!hit){let md=Infinity;for(const obj of detectedObjects){const[x1,y1,x2,y2]=obj.bbox;const d=Math.hypot(ox-(x1+x2)/2,oy-(y1+y2)/2);if(d<md){md=d;hit=obj;}}if(md>300){hideObjPopup();return;}}
  if(!video.paused)video.pause();
  const sRect=document.getElementById('edPreview').getBoundingClientRect();
  showObjPopup(e.clientX-sRect.left,e.clientY-sRect.top,hit);selectObjectForPanel(hit);
}
function showObjPopup(x,y,obj){
  const pop=document.getElementById('objPopup');
  document.getElementById('popupName').textContent=obj.label;document.getElementById('popupConf').textContent=Math.round(obj.confidence*100)+'%';
  pop.style.display='block';
  const stage=document.getElementById('edPreview');let px=x+14,py=y-14;
  if(px+180>stage.offsetWidth)px=x-194;if(py+90>stage.offsetHeight)py=y-90;
  pop.style.left=Math.max(4,px)+'px';pop.style.top=Math.max(4,py)+'px';
}
function hideObjPopup(){document.getElementById('objPopup').style.display='none';}
function pickAndClose(action){setEditorAction(action);hideObjPopup();}

function selectObjectForPanel(obj){
  selectedObject=obj;
  document.getElementById('selName').textContent=obj.label;document.getElementById('selConf').textContent=Math.round(obj.confidence*100)+'% confidence';
  document.getElementById('selIcon').textContent=obj.label.charAt(0).toUpperCase();
  document.getElementById('epHint').style.display='none';document.getElementById('epSelSec').style.display='';
  const v=document.getElementById('mainVideo');const cur=Math.round(v.currentTime*10)/10;
  document.getElementById('startTime').value=Math.max(0,cur).toFixed(1);
  document.getElementById('endTime').value=Math.min(Math.floor(v.duration),+cur+5).toFixed(0);
  syncRangeFromInputs();
  document.querySelectorAll('.obj-chip').forEach(c=>c.classList.toggle('on',c.dataset.label===obj.label));
  showToast('Selected '+obj.label+' — drag the red handles on the timeline or type the time range','info');
}
function clearSelection(){
  selectedObject=null;hideObjPopup();
  document.getElementById('epHint').style.display='flex';document.getElementById('epSelSec').style.display='none';
  document.querySelectorAll('.obj-chip').forEach(c=>c.classList.remove('on'));
}
function setEditorAction(a){
  currentAction=a;document.querySelectorAll('.action-card').forEach(c=>c.classList.remove('on'));
  document.getElementById('ac-'+a)?.classList.add('on');
}

function setTool(t){
  _currentTool=t;
  document.getElementById('edBtnRemove').classList.toggle('on',t==='remove');
  document.getElementById('edBtnTrim').classList.toggle('on',t==='trim');
  document.getElementById('panelRemove').style.display=t==='remove'?'flex':'none';
  document.getElementById('panelTrim').style.display=t==='trim'?'flex':'none';
  document.getElementById('tlRemoveZone').style.display=t==='remove'?'':'none';
  document.getElementById('tlEditL').style.display=t==='remove'?'':'none';
  document.getElementById('tlEditR').style.display=t==='remove'?'':'none';
  updateTimeline();
}

function togglePlay(){
  const v=document.getElementById('mainVideo'),btn=document.getElementById('playBtn');
  if(v.paused){v.play();btn.innerHTML='&#9646;&#9646;';}else{v.pause();btn.innerHTML='&#9654;';}
}

function buildRuler(){
  const ruler=document.getElementById('tlRuler');ruler.innerHTML='';
  const dur=videoDuration||1;const steps=Math.min(10,Math.floor(dur));
  for(let i=0;i<=steps;i++){const t=Math.round(dur*i/steps);const d=document.createElement('span');d.className='tl-tick';d.textContent=fmtTime(t);ruler.appendChild(d);}
}

function updateTimeline(){
  document.getElementById('tlGrayL').style.width=(_trimS*100)+'%';
  document.getElementById('tlGrayR').style.width=((1-_trimE)*100)+'%';
  document.getElementById('tlTrimL').style.left=(_trimS*100)+'%';
  document.getElementById('tlTrimR').style.left=(_trimE*100)+'%';
  const es=_editS*100,ew=(_editE-_editS)*100;
  const rz=document.getElementById('tlRemoveZone');
  if(_editE>_editS){rz.style.left=es+'%';rz.style.width=ew+'%';}
  document.getElementById('tlEditL').style.left=(_editS*100)+'%';
  document.getElementById('tlEditR').style.left=(_editE*100)+'%';
  const dur=videoDuration||1;
  document.getElementById('trimStart').value=(_trimS*dur).toFixed(1);
  document.getElementById('trimEnd').value=(_trimE*dur).toFixed(1);
  document.getElementById('trimDurLbl').textContent='Duration: '+((_trimE-_trimS)*dur).toFixed(1)+'s';
}

function syncRangeFromInputs(){
  const dur=videoDuration||1;
  const s=parseFloat(document.getElementById('startTime').value)||0;
  const en=parseFloat(document.getElementById('endTime').value)||0;
  _editS=Math.max(0,Math.min(s/dur,1));_editE=Math.max(_editS,Math.min(en/dur,1));
  updateTimeline();
}
function syncTrimFromInputs(){
  const dur=videoDuration||1;
  const s=parseFloat(document.getElementById('trimStart').value)||0;
  const en=parseFloat(document.getElementById('trimEnd').value)||dur;
  _trimS=Math.max(0,Math.min(s/dur,1));_trimE=Math.max(_trimS,Math.min(en/dur,1));
  updateTimeline();
}
function updateRangeHl(){syncRangeFromInputs();}

function tlXtoFrac(e){const track=document.getElementById('tlTrack');const rect=track.getBoundingClientRect();return Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width));}
function tlDown(e){
  const f=tlXtoFrac(e);
  const handles=[{id:'trim-l',pos:_trimS,on:true},{id:'trim-r',pos:_trimE,on:true},{id:'edit-l',pos:_editS,on:_currentTool==='remove'},{id:'edit-r',pos:_editE,on:_currentTool==='remove'}];
  let best=null,bd=0.05;
  handles.forEach(h=>{if(h.on&&Math.abs(h.pos-f)<bd){bd=Math.abs(h.pos-f);best=h.id;}});
  if(best){_dragging=best;}else{const v=document.getElementById('mainVideo');if(v.duration)v.currentTime=f*v.duration;}
}
function tlMove(e){
  if(!_dragging)return;const f=tlXtoFrac(e);const dur=videoDuration||1;
  if(_dragging==='trim-l'){_trimS=Math.min(f,_trimE-0.01);}
  else if(_dragging==='trim-r'){_trimE=Math.max(f,_trimS+0.01);}
  else if(_dragging==='edit-l'){_editS=Math.min(f,_editE-0.01);document.getElementById('startTime').value=(_editS*dur).toFixed(1);}
  else if(_dragging==='edit-r'){_editE=Math.max(f,_editS+0.01);document.getElementById('endTime').value=(_editE*dur).toFixed(1);}
  updateTimeline();
}
function tlUp(){_dragging=null;}

function setRangeFromNow(dur){
  const v=document.getElementById('mainVideo');const cur=Math.round(v.currentTime*10)/10;
  document.getElementById('startTime').value=Math.max(0,cur).toFixed(1);
  document.getElementById('endTime').value=Math.min(Math.floor(v.duration),+cur+dur).toFixed(0);
  syncRangeFromInputs();
}
function setFullRange(){
  const v=document.getElementById('mainVideo');
  document.getElementById('startTime').value='0';document.getElementById('endTime').value=Math.floor(v.duration);
  syncRangeFromInputs();
}

async function runDetection(){
  if(!currentFile){showToast('Upload a video first','error');return;}
  const chips=document.getElementById('objChips');
  chips.innerHTML='<div class="skeleton" style="height:28px;width:80px;border-radius:100px;display:inline-block;margin-right:4px;"></div><div class="skeleton" style="height:28px;width:70px;border-radius:100px;display:inline-block;"></div>';
  try{
    const r=await fetch(API+'/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentFile.name})});
    const d=await r.json();detectedObjects=d.objects||[];chips.innerHTML='';
    if(!detectedObjects.length){chips.innerHTML='<span style="font-size:12px;color:var(--t3);">No objects found</span>';return;}
    detectedObjects.forEach(obj=>{
      const chip=document.createElement('button');chip.className='obj-chip';chip.dataset.label=obj.label;
      chip.innerHTML=obj.label+'<span class="obj-chip-conf"> '+Math.round(obj.confidence*100)+'%</span>';
      chip.onclick=()=>{const v=document.getElementById('mainVideo');if(!v.paused)v.pause();selectObjectForPanel(obj);};
      chips.appendChild(chip);
    });
    showToast(detectedObjects.length+' objects found — click one to select','success');
  }catch(err){chips.innerHTML='<span style="font-size:12px;color:var(--r);">Detection failed</span>';}
}

function edProcess(){if(_currentTool==='remove')processEdit();else processTrim();}

async function processEdit(){
  if(!currentFile){showToast('Upload a video first','error');return;}
  if(!selectedObject){showToast('Click an object in the video to select it','error');return;}
  const s=parseFloat(document.getElementById('startTime').value)||0;
  const en=parseFloat(document.getElementById('endTime').value)||10;
  if(en<=s){showToast('End time must be after start time','error');return;}
  document.getElementById('epProgress').style.display='';document.getElementById('epResult').style.display='none';
  document.getElementById('epProgBar').style.width='0%';
  const btn=document.getElementById('edProcBtn');btn.disabled=true;btn.classList.add('running');btn.textContent='Processing...';
  document.getElementById('procBtn').disabled=true;
  try{
    const r=await fetch(API+'/process-edit',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({filename:currentFile.name,text_command:currentAction+' the '+selectedObject.label,start_time:s,end_time:en,bbox:selectedObject.bbox})});
    const d=await r.json();currentJobId=d.job_id;pollJob();
  }catch(err){
    btn.disabled=false;btn.classList.remove('running');btn.textContent='Process';
    document.getElementById('procBtn').disabled=false;document.getElementById('epProgress').style.display='none';
    showToast('Processing failed','error');
  }
}

function pollJob(){
  if(pollInterval)clearInterval(pollInterval);
  const labels={'5':'Extracting frames...','20':'Generating masks...','30':'AI removing object...','72':'Finalising...','80':'Assembling video...','90':'Adding audio...','100':'Complete!'};
  pollInterval=setInterval(async()=>{
    try{
      const r=await fetch(API+'/status/'+currentJobId);const d=await r.json();
      const pct=d.progress||0;document.getElementById('epProgBar').style.width=pct+'%';
      const k=Object.keys(labels).reverse().find(k=>pct>=+k);if(k)document.getElementById('epProgLabel').textContent=labels[k];
      if(d.status==='completed'){clearInterval(pollInterval);showResult();}
      else if(d.status==='failed'){
        clearInterval(pollInterval);const btn=document.getElementById('edProcBtn');
        btn.disabled=false;btn.classList.remove('running');btn.textContent='Process';
        document.getElementById('procBtn').disabled=false;document.getElementById('epProgress').style.display='none';
        showToast('Failed: '+(d.error||'Unknown error'),'error');
      }
    }catch(e){}
  },3000);
}

function showResult(){
  document.getElementById('epProgress').style.display='none';document.getElementById('epResult').style.display='';
  const btn=document.getElementById('edProcBtn');btn.disabled=false;btn.classList.remove('running');btn.textContent='Process';
  document.getElementById('procBtn').disabled=false;document.getElementById('edDlBtn').classList.add('show');
  showToast('Done! Object removed from full video.','success');
  saveHistory({jobId:currentJobId,filename:currentFile.name,action:currentAction,object:selectedObject?.label||null,ts:Date.now()});
  refreshDashboard();
}

async function processTrim(){
  if(!currentFile){showToast('Upload a video first','error');return;}
  const dur=videoDuration||1;const s=_trimS*dur,en=_trimE*dur;
  if(en-s<0.5){showToast('Select at least 0.5s to trim','error');return;}
  document.getElementById('trimProgress').style.display='';document.getElementById('trimResult').style.display='none';
  const btn=document.getElementById('trimProcBtn');btn.disabled=true;
  try{
    const r=await fetch(API+'/api/trim-video',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentFile.name,trim_start:s,trim_end:en})});
    const d=await r.json();
    if(d.job_id){currentJobId=d.job_id;document.getElementById('trimProgress').style.display='none';document.getElementById('trimResult').style.display='';document.getElementById('edDlBtn').classList.add('show');showToast('Trim complete!','success');}
    else{showToast('Trim failed: '+(d.error||'Unknown'),'error');document.getElementById('trimProgress').style.display='none';}
  }catch(err){document.getElementById('trimProgress').style.display='none';showToast('Trim failed','error');}
  finally{btn.disabled=false;}
}

function downloadResult(){if(currentJobId)window.open(API+'/download/'+currentJobId,'_blank');}
function clearResult(){document.getElementById('epResult').style.display='none';document.getElementById('trimResult').style.display='none';document.getElementById('edDlBtn').classList.remove('show');currentJobId=null;}

function resetEditor(){
  currentFile=null;currentJobId=null;selectedObject=null;detectedObjects=[];
  if(pollInterval)clearInterval(pollInterval);stopCanvasLoop();
  document.getElementById('edEmpty').style.display='flex';document.getElementById('edApp').style.display='none';
  document.getElementById('videoInput').value='';document.getElementById('mainVideo').src='';
  document.getElementById('objChips').innerHTML='';document.getElementById('edDlBtn').classList.remove('show');
  document.getElementById('epHint').style.display='flex';document.getElementById('epSelSec').style.display='none';
  document.getElementById('epProgress').style.display='none';document.getElementById('epResult').style.display='none';
  hideObjPopup();clearSelection();
}
"""

js_s = text.find('\nfunction handleDragOver(')
js_e = text.find('\n/* -- COMMAND PALETTE', js_s)
print(f"JS replace: {js_s}..{js_e}")
text = text[:js_s] + '\n' + new_js.strip() + '\n\n' + text[js_e:]

with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

s2=text.find('<script>');e2=text.rfind('</script>');js=text[s2+8:e2]
bopen=js.count('{');bclose=js.count('}')
print(f"Done. Lines: {len(text.splitlines())}")
print(f"Brace balance: {bopen}/{bclose} diff={bopen-bclose}")
must=['uploadFile','handleStageClick','startCanvasLoop','setTool','updateTimeline','tlDown','processTrim','processEdit','pollJob','showResult','resetEditor','showToast','fmtTime','generateImage','openCmd']
for fn in must:
    print(f"  {'OK' if fn in js else 'MISS'} {fn}")

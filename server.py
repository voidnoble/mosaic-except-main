"""Phase 3: 실시간 스트림 + 얼굴 등록 관리 서버 (server.py)

- GET  /             : 등록 얼굴 관리 웹 UI
- GET  /persons      : 등록 인물 목록
- POST /persons      : 인물 등록 (multipart: name + 이미지 파일들)
- DELETE /persons/{name}: 인물 삭제
- WS   /ws/process   : JPEG 프레임 수신 → SCRFD+ArcFace+ByteTrack+마스크 익명화 → JPEG 송신

실행:
  python server.py --db face_db.npz --port 8000
테스트:
  python server.py --db face_db.npz &  # 백그라운드
  # 브라우저: http://localhost:8000  (등록 UI)
  # ws 클라이언트: test_ws_client.py 참고
"""
import argparse
import io
import os
import sys
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anonymize
import prototype_scrfd_arcface as proto
from face_store import FaceStore
from tracking import BYTETracker
from tracking.track import TrackState

app = FastAPI(title="MOFY Phase 3")
_state = {}  # {"face_app": ..., "store": ..., "db_path": ..., "lock": ...}


def get_state():
    return _state


def init(db_path, det_size=640, threshold=0.4, mask_region="head",
         mask_mode="pixelate"):
    _state["face_app"] = proto.get_app(det_size=(det_size, det_size),
                                      providers=["CPUExecutionProvider"])
    _state["store"] = (FaceStore.from_npz(db_path)
                       if os.path.exists(db_path) else FaceStore())
    _state["db_path"] = db_path
    _state["threshold"] = threshold
    _state["mask_region"] = mask_region
    _state["mask_mode"] = mask_mode
    _state["lock"] = threading.Lock()


def save_store():
    st = _state
    with st["lock"]:
        st["store"].save(st["db_path"])


# ---------------- REST: 얼굴 등록 관리 ----------------
@app.get("/persons")
def list_persons():
    return {"persons": _state["store"].list_persons()}


@app.post("/persons")
async def add_person(name: str = Form(...),
                     files: list[UploadFile] = File(...)):
    st = _state
    face_app = st["face_app"]
    embs = []
    for f in files:
        data = await f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        faces = face_app.get(img)
        for fc in faces:
            if fc.det_score >= 0.5 and fc.embedding is not None:
                embs.append(fc.embedding)
    if not embs:
        raise HTTPException(400, "등록 가능한 얼굴을 찾지 못했습니다")
    with st["lock"]:
        st["store"].add_person(name, np.stack(embs))
        st["store"].save(st["db_path"])
    return {"name": name, "n_images": len(embs)}


@app.delete("/persons/{name}")
def delete_person(name: str):
    st = _state
    with st["lock"]:
        ok = st["store"].remove_person(name)
        if ok:
            st["store"].save(st["db_path"])
    if not ok:
        raise HTTPException(404, f"등록되지 않은 인물: {name}")
    return {"deleted": name}


# ---------------- WebSocket: 실시간 스트림 처리 ----------------
def process_frame(img, tracker, track_lm):
    st = _state
    face_app, store = st["face_app"], st["store"]
    faces = face_app.get(img)
    dets = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0:
            dets.append((np.array([x1, y1, w, h], dtype=np.float32),
                         float(f.det_score)))
    tracks = tracker.update(dets)

    # 검출↔트랙 매칭 (간단 IoU 탐욕)
    from tracking.matching import iou_batch
    tracked = [t for t in tracks if t.state == TrackState.Tracked]
    fboxes = np.asarray([f.bbox for f in faces], dtype=np.float32)
    id2face = {}
    if tracked and len(faces):
        tboxes = np.asarray([t.render_tlbr for t in tracked], dtype=np.float32)
        ious = iou_batch(tboxes, fboxes)
        used = set()
        for ti, t in enumerate(tracked):
            fi = int(np.argmax(ious[ti]))
            if ious[ti, fi] >= 0.3 and fi not in used:
                id2face[t.track_id] = faces[fi]
                used.add(fi)

    for t in tracks:
        if t.state != TrackState.Tracked or t.identity_locked:
            continue
        f = id2face.get(t.track_id)
        if f is None:
            continue
        lm = getattr(f, "landmark_2d_106", None)
        if lm is not None:
            track_lm[t.track_id] = np.asarray(lm, dtype=np.float32)
        x1, y1, x2, y2 = f.bbox
        if (f.det_score >= 0.5 and (x2 - x1) >= 20
                and f.embedding is not None):
            name, _ = store.search(f.embedding, st["threshold"])
        else:
            name = "unknown"  # 품질 미달 → fail-closed
        t.vote(name)

    regions, draw_faces, results = [], [], []
    for t in tracks:
        if t.state not in (TrackState.Tracked, TrackState.Lost):
            continue
        ident = t.identity or "unknown"
        bb = tuple(t.render_tlbr)
        if ident == "unknown":
            regions.append({"bbox": bb, "lm": track_lm.get(t.track_id)})
        draw_faces.append(type("F", (), {"bbox": np.array(bb)})())
        results.append((ident, 0.0))
    out = anonymize.anonymize(img, regions, region=st["mask_region"],
                              mode=st["mask_mode"])
    return proto.annotate(out, draw_faces, results)


@app.websocket("/ws/process")
async def ws_process(ws: WebSocket):
    await ws.accept()
    tracker = BYTETracker(track_thresh=0.5, max_time_lost=30,
                          ema_alpha=0.6, vote_to_lock=3)
    track_lm = {}
    try:
        while True:
            data = await ws.receive_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            out = process_frame(img, tracker, track_lm)
            _, buf = cv2.imencode(".jpg", out,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            await ws.send_bytes(buf.tobytes())
    except Exception:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------- 등록 UI ----------------
INDEX_HTML = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>MOFY 얼굴 등록</title>
<style>body{font-family:sans-serif;max-width:640px;margin:2em auto;padding:0 1em}
li{margin:.4em 0}button{margin-left:.5em}</style></head><body>
<h2>등록 얼굴 관리</h2>
<form id="f"><input name="name" placeholder="이름" required>
<input type="file" name="files" multiple accept="image/*" required>
<button>등록</button></form>
<h3>등록된 인물</h3><ul id="list"></ul>
<script>
async function refresh(){const r=await fetch('/persons');const d=await r.json();
 document.getElementById('list').innerHTML=d.persons.map(n=>
 `<li>${n}<button onclick="del('${n}')">삭제</button></li>`).join('')||'<li>(없음)</li>'}
async function del(n){await fetch('/persons/'+encodeURIComponent(n),{method:'DELETE'});refresh()}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
 const r=await fetch('/persons',{method:'POST',body:new FormData(e.target)});
 alert(r.ok?'등록 완료':await r.text());e.target.reset();refresh()}
refresh()</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def main():
    import uvicorn
    p = argparse.ArgumentParser(description="MOFY Phase 3 서버")
    p.add_argument("--db", default="face_db_server.npz")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--mask-region", default="head",
                   choices=["bbox", "face", "head"])
    p.add_argument("--mask-mode", default="pixelate",
                   choices=["pixelate", "blur", "solid"])
    a = p.parse_args()
    init(a.db, det_size=a.det_size, threshold=a.threshold,
         mask_region=a.mask_region, mask_mode=a.mask_mode)
    print(f"서버 시작: http://localhost:{a.port}  (DB: {a.db})")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

"""WebRTC 검증 (test_webrtc.py).

환경 제한: 이 VM 샌드박스는 UDP를 차단하므로(aiortc 미디어 전송은 UDP 필수)
실제 미디어 루프백은 불가. 대신 검증 가능한 두 부분을 테스트:
  A. 시그널링: offer → answer 200, 유효 SDP + ICE candidate 포함
  B. 미디어 처리 경로: WebRTC 핸들러와 동일한 process_frame 호출로
     프레임이 익명화 처리되는지 검증 (ProcessedTrack.recv 내부 로직과 동일)
"""
import asyncio
import sys

import cv2
import httpx
import numpy as np
from aiortc import RTCPeerConnection

BASE = "http://localhost:8001"
ok = True


def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f" ({extra})" if extra else ""))
    ok = ok and cond


async def signaling_test():
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    async with httpx.AsyncClient(trust_env=False, timeout=60) as c:
        r = await c.post(f"{BASE}/webrtc/offer", json={
            "sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    check("A. offer→answer", r.status_code == 200, f"HTTP {r.status_code}")
    ans = r.json()
    check("A. answer SDP 유효",
          ans["type"] == "answer" and "candidate" in ans["sdp"],
          f"sdp {len(ans['sdp'])}자")
    await pc.close()


def media_path_test():
    # server.process_frame을 WebRTC 핸들러와 동일한 조건으로 호출
    sys.path.insert(0, ".")
    import server
    server.init("/tmp/p3_db.npz")  # 테스트 프로세스에서 모델 초기화
    from tracking import BYTETracker
    tracker = BYTETracker(track_thresh=0.5, max_time_lost=30,
                          ema_alpha=0.6, vote_to_lock=3)
    track_lm = {}
    cap = cv2.VideoCapture("/tmp/static_test.mp4")
    diffs = []
    for _ in range(6):
        ret, img = cap.read()
        if not ret:
            break
        out = server.process_frame(img, tracker, track_lm)
        diffs.append(np.abs(out.astype(int) - img.astype(int)).mean())
    check("B. 처리 경로 동작", len(diffs) == 6, f"{len(diffs)}프레임")
    check("B. 익명화 적용됨", np.mean(diffs) > 1.0,
          f"평균 차이 {np.mean(diffs):.1f}")
    cv2.imwrite("/tmp/webrtc_out.jpg", out)


asyncio.run(signaling_test())
media_path_test()
sys.exit(0 if ok else 1)

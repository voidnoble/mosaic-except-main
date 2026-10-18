"""Phase 3 서버 E2E 테스트: REST 등록 + WebSocket 스트림 처리."""
import asyncio
import sys

import cv2
import httpx
_client = httpx.Client(trust_env=False)
import numpy as np
import websockets

BASE = "http://localhost:8000"
WS = "ws://localhost:8000/ws/process"

ok = True


def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f" ({extra})" if extra else ""))
    ok = ok and cond


# 1) 등록
imgs = ["data/test_images/bradley_cooper/%s" % f for f in
        __import__("os").listdir("data/test_images/bradley_cooper")[:2]]
files = [("files", (f.split("/")[-1], open(f, "rb"), "image/jpeg")) for f in imgs]
r = _client.post(f"{BASE}/persons", data={"name": "bradley"},
               files=files)
check("POST /persons", r.status_code == 200, r.text[:80])
for _, (_, fh, _) in files:
    fh.close()

r = _client.get(f"{BASE}/persons")
check("GET /persons", "bradley" in r.json()["persons"], str(r.json()))


async def ws_test():
    async with websockets.connect(WS, max_size=20_000_000) as ws:
        img = cv2.imread("data/multiface.jpg")
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        for i in range(3):
            await ws.send(buf.tobytes())
            data = await asyncio.wait_for(ws.recv(), timeout=120)
            out = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            check(f"WS frame {i}", out is not None and out.shape == img.shape,
                  f"shape={None if out is None else out.shape}")
            if i == 2:
                cv2.imwrite("/tmp/ws_out.jpg", out)


asyncio.run(ws_test())

# 4) 삭제
r = _client.delete(f"{BASE}/persons/bradley")
check("DELETE /persons/bradley", r.status_code == 200, r.text[:60])
r = _client.get(f"{BASE}/persons")
check("삭제 후 목록", r.json()["persons"] == [], str(r.json()))

sys.exit(0 if ok else 1)

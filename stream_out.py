"""Phase 3 stretch: RTMP 스트림 출력 (stream_out.py)

pipeline_v2 --output rtmp://host:1935/app/key 로 주면
처리된 프레임을 ffmpeg 파이프로 실시간 송출한다.
인터페이스는 cv2.VideoWriter와 동일 (write/release).
"""
import shutil
import subprocess


class RTMPWriter:
    def __init__(self, url, fps, w, h):
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("ffmpeg이 필요합니다 (RTMP 송출)")
        self.proc = subprocess.Popen(
            [ff, "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps:.2f}",
             "-i", "-",
             "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
             "-f", "flv", url],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._w, self._h = w, h

    def write(self, frame):
        if frame.shape[1] != self._w or frame.shape[0] != self._h:
            import cv2
            frame = cv2.resize(frame, (self._w, self._h))
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def release(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=10)

    def isOpened(self):
        return self.proc.poll() is None


def make_writer(output, fps, w, h):
    """rtmp:// 로 시작하면 RTMPWriter, 아니면 cv2.VideoWriter(mp4v)."""
    if output.startswith("rtmp://"):
        return RTMPWriter(output, fps, w, h)
    import cv2
    return cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                           fps, (w, h))

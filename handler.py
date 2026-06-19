#!/usr/bin/env python3
# Gempronun — RunPod Serverless handler.
# RunPod gọi handler(job) với job["input"] là JSON (KHÔNG có multipart upload),
# nên audio truyền dưới dạng base64. Ta tái dùng NGUYÊN logic Flask trong app.py
# bằng app.test_client() — giữ y hệt /score /grade /grade_ph /transcribe /pron.
#
# Input job:
#   {
#     "input": {
#       "route":     "score" | "grade" | "grade_ph" | "transcribe" | "pron" | "health",
#       "audio_b64": "<base64 của file audio webm/mp3/wav/m4a...>",   # bắt buộc (trừ health)
#       "filename":  "audio.webm",   # tùy chọn, chỉ để ffmpeg đoán định dạng
#       "text":      "...",          # văn bản mẫu (score/grade/grade_ph/pron)
#       "words":     "...",          # gợi ý từ (score) — tùy chọn
#       "lang":      "en",           # tùy chọn
#       "prompt":    "...",          # initial_prompt cho whisper — tùy chọn
#       "fast":      true            # transcribe nhanh — tùy chọn
#     }
#   }
# Output: y hệt JSON mà route Flask tương ứng trả về.
import base64
import io
import os

# Mặc định chạy GPU, không bật API key nội bộ (RunPod lo auth ở rìa qua Bearer token).
os.environ.setdefault("ASR_DEVICE", "cuda")
os.environ.setdefault("ASR_COMPUTE", "float16")
os.environ.setdefault("ASR_API_KEY", "")
os.environ.setdefault("ASR_MODEL", "medium.en")
os.environ.setdefault("HF_HOME", "/models")

import app as asrapp  # import = nạp model 1 lần (cold start)
import runpod

_client = asrapp.app.test_client()

ROUTE_MAP = {
    "score": "/score",
    "grade": "/grade",
    "grade_ph": "/grade_ph",
    "transcribe": "/transcribe",
    "pron": "/pron",
    "health": "/health",
}


def handler(job):
    inp = job.get("input") or {}
    route = (inp.get("route") or "score").strip()
    path = ROUTE_MAP.get(route)
    if not path:
        return {"ok": False, "err": f"unknown route '{route}'"}

    if route == "health":
        return _client.get("/health").get_json()

    b64 = inp.get("audio_b64") or inp.get("audio")
    if not b64:
        return {"ok": False, "err": "no audio (cần 'audio_b64')"}
    try:
        audio = base64.b64decode(b64)
    except Exception as e:
        return {"ok": False, "err": f"base64 decode failed: {e}"}
    if not audio:
        return {"ok": False, "err": "empty audio"}

    data = {"file": (io.BytesIO(audio), inp.get("filename", "audio.webm"))}
    for k in ("text", "words", "lang", "prompt", "fast"):
        v = inp.get(k)
        if v is not None:
            data[k] = ("1" if v is True else "0" if v is False else str(v))

    resp = _client.post(path, data=data, content_type="multipart/form-data")
    out = resp.get_json(silent=True)
    if out is None:
        return {"ok": False, "err": "non-json response", "status": resp.status_code}
    return out


def _warmup():
    """Chạy 1 inference GIẢ lúc worker khởi động để 'làm nóng' GPU kernel (cuDNN/CTranslate2),
    tránh lượt chấm THẬT đầu tiên sau cold-start bị lỗi/null."""
    try:
        import io as _io, wave, struct, math
        sr = 16000
        n = int(sr * 1.3)
        b = _io.BytesIO()
        wf = wave.open(b, "wb")
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(b"".join(struct.pack("<h", int(2500 * math.sin(2 * math.pi * 200 * i / sr))) for i in range(n)))
        wf.close()
        for route, payload in (("/score", {"text": "warm up test sentence"}),):
            d = {"file": (_io.BytesIO(b.getvalue()), "w.wav")}
            d.update(payload)
            _client.post(route, data=d, content_type="multipart/form-data")
        print("[warmup] scoring path warmed (GPU kernels ready)", flush=True)
    except Exception as e:
        print("[warmup] skipped:", e, flush=True)


if __name__ == "__main__":
    _warmup()
    runpod.serverless.start({"handler": handler})

#!/usr/bin/env python3
# Ví dụ client gọi RunPod Serverless endpoint (cách samio.fun sẽ gọi).
# RunPod KHÁC IP cố định: gọi qua https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync
# với header Authorization: Bearer <RUNPOD_API_KEY> và body {"input": {...}}.
import base64, os, sys, requests

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "2gmiijirwfd90y")
API_KEY     = os.environ["RUNPOD_API_KEY"]
audio_path  = sys.argv[1]
text        = sys.argv[2] if len(sys.argv) > 2 else "apple"
route       = sys.argv[3] if len(sys.argv) > 3 else "score"

with open(audio_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# /runsync = đồng bộ (chờ kết quả). Có thể dùng /run (bất đồng bộ) + /status/<id> nếu cần.
r = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"input": {"route": route, "text": text,
                    "filename": os.path.basename(audio_path), "audio_b64": b64}},
    timeout=120,
)
print(r.status_code)
print(r.json())   # {"id":..., "status":"COMPLETED", "output": { ...kết quả app.py... }}

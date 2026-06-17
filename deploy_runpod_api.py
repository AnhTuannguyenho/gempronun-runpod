#!/usr/bin/env python3
# Cập nhật image cho RunPod Serverless endpoint qua REST API.
# Cơ chế: image nằm trên TEMPLATE; endpoint trỏ templateId. PATCH template -> RunPod
# tự rolling-release endpoint.
#
# Dùng:
#   export RUNPOD_API_KEY=...                     # bắt buộc
#   export IMAGE=docker.io/<user>/gempronun-runpod:1   # bắt buộc (image đã push)
#   python deploy_runpod_api.py [endpoint_id]     # mặc định 4tzogl7txqk2ax
#
# Cờ phụ (env): DISK_GB (mặc định 20), DRY_RUN=1 (chỉ in, không sửa).
import json
import os
import sys
import urllib.request

API = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY") or sys.exit("Thiếu RUNPOD_API_KEY")
IMAGE = os.environ.get("IMAGE") or sys.exit("Thiếu IMAGE (vd docker.io/you/gempronun-runpod:1)")
EP = sys.argv[1] if len(sys.argv) > 1 else "4tzogl7txqk2ax"
DISK_GB = int(os.environ.get("DISK_GB", "20"))
DRY = os.environ.get("DRY_RUN") == "1"

HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, headers=HDR, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read().decode()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}


# 1) Đọc endpoint + template hiện tại
st, ep = req("GET", f"/endpoints/{EP}?includeTemplate=true")
if st != 200:
    sys.exit(f"GET endpoint lỗi {st}: {ep}")
tid = ep.get("templateId")
tpl = ep.get("template") or {}
print(f"Endpoint : {ep.get('name')} ({EP})")
print(f"Template : {tid}  image hiện tại = {tpl.get('imageName')!r}")
print(f"Workers  : min={ep.get('workersMin')} max={ep.get('workersMax')}  GPU={ep.get('gpuTypeIds')}")
if not tid:
    sys.exit("Endpoint chưa gắn template — cần tạo template trước (báo lại để xử lý).")

# 2) PATCH template -> image mới (giữ env cũ, đảm bảo các biến engine)
env = dict(tpl.get("env") or {})
env.setdefault("ASR_DEVICE", "cuda")
env.setdefault("ASR_COMPUTE", "float16")
env.setdefault("ASR_MODEL", "medium.en")
env.setdefault("HF_HOME", "/models")

patch = {"imageName": IMAGE, "env": env, "containerDiskInGb": DISK_GB}
print(f"\n==> PATCH template {tid}:")
print(json.dumps(patch, indent=2, ensure_ascii=False))
if DRY:
    print("\n(DRY_RUN=1 — không gửi)")
    sys.exit(0)

st, out = req("PATCH", f"/templates/{tid}", patch)
if st not in (200, 201):
    sys.exit(f"\nPATCH template lỗi {st}: {out}")
print(f"\n✅ Đã cập nhật image. RunPod sẽ rolling-release endpoint {EP}.")
print("   Theo dõi worker ở console -> endpoint -> Workers/Logs cho tới khi image pull xong.")

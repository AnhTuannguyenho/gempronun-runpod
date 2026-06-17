# Gempronun — RunPod Serverless

Engine chấm phát âm (faster-whisper + wav2vec2 GOP, GPU) đóng gói thành **RunPod Serverless worker**.
Khác bản vast.ai (`../gempronun-serverless`, PyWorker proxy HTTP): RunPod gọi `handler(job)` với
`job["input"]` là **JSON**, audio truyền **base64**. Model **nướng sẵn trong image** → cold-start nhanh.

## Thành phần
- `app.py` — engine Flask gốc (giữ NGUYÊN, routes `/score /grade /grade_ph /transcribe /pron /health`).
- `handler.py` — RunPod handler; tái dùng app.py qua `test_client()`, dispatch theo `input.route`.
- `Dockerfile` — base CUDA 12.4, cài deps + torch cu124 + tải sẵn model vào `/models`.
- `deploy.sh` — build + push image (chạy trên máy/VPS có Docker).
- `deploy_runpod_api.py` — PATCH template của endpoint sang image mới (rolling release).
- `client_example.py` / `make_test_input.py` — test.

## Định dạng input
```json
{ "input": {
    "route": "score",            // score | grade | grade_ph | transcribe | pron | health
    "text": "apple",             // văn bản mẫu
    "audio_b64": "<base64>",     // file audio (webm/mp3/wav/m4a...)
    "filename": "audio.webm",    // tùy chọn
    "words": "", "lang": "en", "prompt": "", "fast": true   // tùy chọn
} }
```
Output = đúng JSON route Flask trả (vd `/score`: `{ok, score, status, band, heard, phones, ...}`).

## Triển khai (3 bước)

### 1) Build + push image (trên máy/VPS có Docker)
```bash
docker login                       # Docker Hub (hoặc: docker login ghcr.io)
IMAGE=docker.io/<user>/gempronun-runpod:1 ./deploy.sh
# Mac ARM build cho RunPod (x86): deploy.sh đã ép --platform linux/amd64
```
> Image ~8–10GB (CUDA + torch + 2 model). Build lần đầu vài phút + tải model.

### 2) Gắn image vào endpoint
Tự động qua API (nhanh nhất):
```bash
export RUNPOD_API_KEY=...                         # Settings → API Keys
export IMAGE=docker.io/<user>/gempronun-runpod:1
python deploy_runpod_api.py 4tzogl7txqk2ax        # đọc template + PATCH image
```
Hoặc thủ công: Console → endpoint → template → **Container Image** = image trên → Save (rolling release).

**Cấu hình endpoint nên đặt:**
- GPU: 16–24GB (RTX 3090/4090/A4000…) — đủ cho medium.en + wav2vec2.
- Container Disk: ≥ 20GB. Workers: min 0 (scale-to-zero), max theo nhu cầu.
- KHÔNG cần Network Volume (model nằm trong image).

### 3) Gọi từ client / samio.fun
```bash
RUNPOD_API_KEY=... python client_example.py path/to/test.wav "apple" score
```
- URL đồng bộ: `POST https://api.runpod.ai/v2/4tzogl7txqk2ax/runsync`
- Header: `Authorization: Bearer <RUNPOD_API_KEY>`
- Body: `{"input": {...}}` → trả `{"status":"COMPLETED","output":{...}}`

## Test local (tùy chọn, cần Docker + GPU)
```bash
python make_test_input.py test.wav "apple" score    # tạo test_input.json
docker run --rm --gpus all -v $PWD/test_input.json:/app/test_input.json \
  docker.io/<user>/gempronun-runpod:1 python -u handler.py   # RunPod chạy test_input.json khi không có job thật
```

## Đấu vào samio.fun
Thêm provider mới (vd `gpronrunpod`): khác `gempronun` ở chỗ gọi RunPod `/v2/<id>/runsync`
với Bearer token + body JSON base64, thay vì POST multipart tới IP cố định. Map output `/score`
giữ nguyên field. (Xem memory [[vastai-asr-gpu]], [[samio-fun-moodle]].)

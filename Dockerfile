# Gempronun — RunPod Serverless image (GPU).
# Model được NƯỚNG SẴN vào image (HF_HOME=/models) -> cold-start chỉ nạp lên VRAM,
# không tải mạng, không phụ thuộc volume.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PIP_NO_CACHE_DIR=1

# System deps: python, ffmpeg (decode audio), espeak-ng (phonemizer), libsndfile (soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg espeak-ng libsndfile1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch CUDA 12.4 (cài riêng từ index pytorch để khớp cu124)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Phần còn lại của engine + RunPod SDK + Kokoro TTS (kèm misaki[en] + spacy en model)
RUN pip install 'numpy<2' faster-whisper transformers flask soundfile phonemizer runpod 'kokoro>=0.9.4' 'misaki[en]' \
    && python -m spacy download en_core_web_sm

COPY app.py handler.py /app/

# Tải sẵn model vào /models (CPU lúc build — không cần GPU). Đổi model qua --build-arg ASR_MODEL=...
ARG ASR_MODEL=medium.en
ENV ASR_MODEL=${ASR_MODEL}
RUN python - <<'PY'
import os
os.environ["HF_HOME"] = "/models"
m = os.environ.get("ASR_MODEL", "medium.en")
from faster_whisper import WhisperModel
WhisperModel(m, device="cpu", compute_type="int8")          # cache whisper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
W = "facebook/wav2vec2-lv-60-espeak-cv-ft"
Wav2Vec2Processor.from_pretrained(W)                         # cache wav2vec2
Wav2Vec2ForCTC.from_pretrained(W)
print("scoring models cached")
# Kokoro TTS: tải model 82M + vài giọng phổ biến vào cache (CPU lúc build)
from kokoro import KPipeline
for lc in ("a", "b"):                                        # US + UK English
    p = KPipeline(lang_code=lc)
    for v in (["af_heart", "am_adam"] if lc == "a" else ["bf_emma", "bm_george"]):
        list(p("hello world", voice=v))                      # trigger tải model+giọng
print("kokoro cached")
PY

ENV ASR_DEVICE=cuda \
    ASR_COMPUTE=float16 \
    ASR_API_KEY=""

CMD ["python", "-u", "handler.py"]

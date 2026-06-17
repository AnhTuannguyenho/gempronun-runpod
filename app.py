#!/usr/bin/env python3
# Samio.fun — ASR + Pronunciation micro-service (tự host, localhost 127.0.0.1:8077)
#  /transcribe : faster-whisper (giọng -> chữ)  — dùng cho nhận dạng từ
#  /pron       : wav2vec2 âm vị + espeak target  — chấm phát âm tới từng âm
import os, subprocess, tempfile, threading, time, warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("ASR_MODEL", "small.en")
PORT       = int(os.environ.get("ASR_PORT", "8077"))
MAX_SEC    = int(os.environ.get("ASR_MAX_SEC", "90"))
MAX_BYTES  = 30 * 1024 * 1024
W2V        = os.environ.get("ASR_W2V", "facebook/wav2vec2-lv-60-espeak-cv-ft")
PH_MIN     = float(os.environ.get("ASR_PH_MIN", "0.45"))  # ngưỡng âm vị coi như "đọc đúng từ"

app = Flask(__name__)

API_KEY = os.environ.get("ASR_API_KEY", "").strip()
CORS_ORIGIN = os.environ.get("ASR_CORS", "*")

@app.before_request
def _auth_gate():
    if request.method == "OPTIONS":
        return ("", 204)
    if request.path == "/health":
        return None
    if API_KEY:
        k = request.headers.get("X-API-Key") or request.args.get("key")
        if k != API_KEY:
            return jsonify(ok=False, err="unauthorized"), 401
    return None

@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = CORS_ORIGIN
    resp.headers["Access-Control-Allow-Headers"] = "X-API-Key, Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp
_lock = threading.Lock()

_DEVICE  = os.environ.get("ASR_DEVICE", "cuda")
_COMPUTE = os.environ.get("ASR_COMPUTE", "float16" if _DEVICE == "cuda" else "int8")
print(f"[asr] loading whisper {MODEL_NAME} on {_DEVICE}/{_COMPUTE}...", flush=True)
if _DEVICE == "cuda":
    _model = WhisperModel(MODEL_NAME, device="cuda", compute_type=_COMPUTE)
else:
    _model = WhisperModel(MODEL_NAME, device="cpu", compute_type=_COMPUTE, cpu_threads=4, num_workers=1)

print(f"[asr] loading phoneme model {W2V}...", flush=True)
import torch, soundfile as sf, numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from phonemizer import phonemize
from phonemizer.separator import Separator
torch.set_num_threads(2)
_w2v_proc = Wav2Vec2Processor.from_pretrained(W2V)
_w2v_model = Wav2Vec2ForCTC.from_pretrained(W2V); _w2v_model.eval()
_w2v_model.to(_DEVICE)
_VOCAB = _w2v_proc.tokenizer.get_vocab()
_ID2TOK = {v: k for k, v in _VOCAB.items()}
_BLANK = _w2v_proc.tokenizer.pad_token_id  # 0
# Hiệu chỉnh sigmoid: posterior thô của giọng đúng ~0.5, sai ~0.0 -> kéo về 0..1 chuẩn.
GOP_P0  = float(os.environ.get("ASR_GOP_P0", "0.18"))  # tâm sigmoid (ngưỡng đúng/sai)
GOP_K   = float(os.environ.get("ASR_GOP_K", "14"))     # độ dốc
# ngưỡng tô màu trên điểm ĐÃ hiệu chỉnh (0..1)
PH_OK   = float(os.environ.get("ASR_PH_OK", "0.60"))
PH_WARN = float(os.environ.get("ASR_PH_WARN", "0.30"))


def _calib(p):
    import math
    return 1.0 / (1.0 + math.exp(-GOP_K * (p - GOP_P0)))
print("[asr] all models ready", flush=True)


def to_wav(src):
    dst = src + ".wav"
    subprocess.run(["ffmpeg", "-y", "-i", src, "-t", str(MAX_SEC), "-ar", "16000", "-ac", "1", "-f", "wav", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    return dst


def _strip_p(p):
    for c in ('ˈ', 'ˌ', 'ː', 'ˑ', 'ʰ', '̩', '̃', 'ʲ'):
        p = p.replace(c, '')
    return p.strip()


def target_phones(text):
    s = phonemize(text, language='en-us', backend='espeak',
                  separator=Separator(phone=' ', word=' | '), strip=True, with_stress=False, njobs=1)
    return [_strip_p(x) for x in s.split() if x != '|' and _strip_p(x)]


def _logits(wav):
    audio, sr = sf.read(wav)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(1)
    iv = _w2v_proc(audio, sampling_rate=16000, return_tensors="pt").input_values.to(_DEVICE)
    with torch.no_grad():
        return _w2v_model(iv).logits[0].cpu()  # [T, V] -> CPU cho phần xử lý sau


def recog_from_logits(logits):
    ids = torch.argmax(logits, dim=-1).unsqueeze(0)
    txt = _w2v_proc.batch_decode(ids)[0]
    return [_strip_p(x) for x in txt.split() if _strip_p(x)]


def recog_phones(wav):
    return recog_from_logits(_logits(wav))


def _ctc_forced_align(logp, tokens, blank=0):
    # Viterbi căn khớp ÉP BUỘC chuỗi âm mục tiêu vào các khung (CTC).
    # logp: [T, V] (numpy log-prob); tokens: list id âm mục tiêu.
    # Trả: ext (chuỗi mở rộng có blank xen kẽ), path (vị trí ext của từng khung).
    T = logp.shape[0]
    L = len(tokens)
    S = 2 * L + 1
    ext = [blank] * S
    for i, tk in enumerate(tokens):
        ext[2 * i + 1] = tk
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.full((T, S), -1, dtype=np.int64)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1:
        dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], s
            if s - 1 >= 0 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], s - 1
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], s - 2
            if best <= NEG:
                continue
            dp[t, s] = best + logp[t, ext[s]]
            bp[t, s] = arg
    s = (S - 2) if (S >= 2 and dp[T - 1, S - 2] > dp[T - 1, S - 1]) else (S - 1)
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t > 0:
            s = int(bp[t, s])
    return ext, path


def _gop_targets(text):
    # Âm mục tiêu cho GOP: GIỮ dấu trường (ː) vì model coi vowel dài là token RIÊNG
    # (vd ɜː khác ɜ). Chỉ bỏ dấu trọng âm. Khớp thẳng vào vocab model.
    s = phonemize(text, language='en-us', backend='espeak',
                  separator=Separator(phone=' ', word=' | '), strip=True, with_stress=False, njobs=1)
    out = []
    for ph in s.split():
        if ph == '|':
            continue
        ph = ph.replace('ˈ', '').replace('ˌ', '').strip()  # bỏ trọng âm, GIỮ ː
        if not ph:
            continue
        if ph in _VOCAB:
            out.append((ph, _VOCAB[ph]))
        else:
            st = _strip_p(ph)  # fallback: bỏ hết dấu phụ
            if st in _VOCAB:
                out.append((st, _VOCAB[st]))
    return out


def _gop_eval(wav, text):
    # GOP THẬT: căn khớp ép buộc + xác suất hậu nghiệm trung bình cho từng âm mục tiêu.
    tg = _gop_targets(text)
    if not tg:
        return {"ok": False, "err": "no target phones"}
    tphones = [p for p, _ in tg]
    toks = [t for _, t in tg]
    logits = _logits(wav)
    said = recog_from_logits(logits)
    logp = torch.log_softmax(logits, dim=-1).numpy()
    T = logp.shape[0]
    if T < len(toks):  # không đủ khung để chứa hết âm
        phones = [{"p": p, "status": "miss"} for p in tphones]
        return {"ok": True, "accuracy": 0, "phones": phones, "said": said}
    ext, path = _ctc_forced_align(logp, toks, blank=_BLANK)
    scores, phones = [], []
    for i, (tk, ph) in enumerate(zip(toks, tphones)):
        frames = [t for t in range(T) if path[t] == 2 * i + 1]
        if frames:
            raw = float(np.exp(np.mean([logp[t, tk] for t in frames])))
            sc = _calib(raw)  # hiệu chỉnh sigmoid
            status = "ok" if sc >= PH_OK else ("warn" if sc >= PH_WARN else "sub")
        else:
            sc, status = 0.0, "miss"
        scores.append(sc)
        phones.append({"p": ph, "status": status, "score": round(sc, 2)})
    acc = round(float(np.mean(scores)) * 100)
    return {"ok": True, "accuracy": acc, "phones": phones, "said": said}


def align_phones(tgt, hyp):
    la, lb = len(tgt), len(hyp)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1): d[i][0] = i
    for j in range(lb + 1): d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if tgt[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    st = ['miss'] * la; i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and tgt[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]: st[i - 1] = 'ok'; i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1: st[i - 1] = 'sub'; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1: st[i - 1] = 'miss'; i -= 1
        else: j -= 1
    ok = sum(1 for s in st if s == 'ok')
    return st, ok, d[la][lb]


@app.get("/health")
def health():
    return jsonify(ok=True, model=MODEL_NAME, w2v=W2V)


def _save_wav(req):
    f = req.files.get("file")
    if not f: return None, (jsonify(ok=False, err="no file"), 400)
    blob = f.read()
    if not blob: return None, (jsonify(ok=False, err="empty"), 400)
    if len(blob) > MAX_BYTES: return None, (jsonify(ok=False, err="too large"), 413)
    tmpd = tempfile.mkdtemp(prefix="asr_")
    src = os.path.join(tmpd, "in")
    with open(src, "wb") as w: w.write(blob)
    return (tmpd, src), None


def _cleanup(paths):
    for p in paths:
        try:
            if p and os.path.exists(p): os.remove(p)
        except Exception: pass


def _whisper(wav, lang="en", hint="", fast=False):
    segments, info = _model.transcribe(
        wav, language=(None if lang == "auto" else lang),
        beam_size=(1 if fast else 5), temperature=0.0, condition_on_previous_text=False,
        initial_prompt=(hint or None), vad_filter=(not fast), word_timestamps=(not fast))
    parts, words, alp, nsp = [], [], [], []
    for seg in segments:
        parts.append(seg.text)
        alp.append(float(getattr(seg, "avg_logprob", 0.0) or 0.0))
        nsp.append(float(getattr(seg, "no_speech_prob", 0.0) or 0.0))
        for wd in (seg.words or []):
            words.append({"w": wd.word.strip(), "start": round(wd.start, 2), "end": round(wd.end, 2)})
    return {"text": "".join(parts).strip(), "words": words,
            "avg_logprob": round((sum(alp)/len(alp)) if alp else -2.0, 3),
            "no_speech": round((max(nsp) if nsp else 1.0), 3),
            "duration": round(getattr(info, "duration", 0.0), 2)}


def _pron_eval(wav, text):
    tgt = target_phones(text)
    hyp = recog_phones(wav)
    if not tgt:
        return {"ok": False, "err": "no target phones"}
    st, ok, dist = align_phones(tgt, hyp)
    denom = max(len(tgt), len(hyp), 1)
    acc = max(0.0, 1 - dist / denom)
    phones = [{"p": tgt[k], "status": st[k]} for k in range(len(tgt))]
    return {"ok": True, "accuracy": round(acc * 100), "target": tgt, "said": hyp, "phones": phones}


def _grade_phonemes(wav, current):
    # GOP thật: đối chiếu THẲNG với âm mục tiêu của CHÍNH từ này (không Whisper,
    # không so với từ khác, không pad 30s).
    r = _gop_eval(wav, current)
    if not r.get("ok"):
        return r
    acc = r["accuracy"] / 100.0
    r["word_ok"] = (acc >= PH_MIN) and bool(r.get("said"))
    return r


@app.post("/transcribe")
def transcribe():
    saved, err = _save_wav(request)
    if err: return err
    tmpd, src = saved
    lang = request.form.get("lang", "en") or "en"
    hint = (request.form.get("prompt", "") or "").strip()[:400]
    fast = (request.form.get("fast", "") or "") in ("1", "true", "yes")
    wav = None
    try:
        wav = to_wav(src)
        t0 = time.time()
        with _lock:
            r = _whisper(wav, lang, hint, fast)
        return jsonify(ok=True, took=round(time.time() - t0, 2), **r)
    except subprocess.CalledProcessError:
        return jsonify(ok=False, err="audio decode failed"), 400
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 500
    finally:
        _cleanup([src, wav]);  os.rmdir(tmpd) if os.path.isdir(tmpd) else None


@app.post("/pron")
def pron():
    saved, err = _save_wav(request)
    if err: return err
    tmpd, src = saved
    text = (request.form.get("text", "") or "").strip()
    wav = None
    try:
        if not text:
            return jsonify(ok=False, err="no text"), 400
        wav = to_wav(src)
        t0 = time.time()
        with _lock:
            r = _pron_eval(wav, text)
        if not r.get("ok"):
            return jsonify(ok=False, err=r.get("err", "pron")), 200
        return jsonify(took=round(time.time() - t0, 2), **r)
    except subprocess.CalledProcessError:
        return jsonify(ok=False, err="audio decode failed"), 400
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 500
    finally:
        _cleanup([src, wav]);  os.rmdir(tmpd) if os.path.isdir(tmpd) else None


@app.post("/grade")
def grade():
    # 1 file -> 1 ffmpeg -> whisper(fast) + âm vị CHẠY SONG SONG -> 1 lần trả
    saved, err = _save_wav(request)
    if err: return err
    tmpd, src = saved
    text = (request.form.get("text", "") or "").strip()
    hint = (request.form.get("prompt", "") or "").strip()[:400]
    lang = request.form.get("lang", "en") or "en"
    wav = None
    try:
        if not text:
            return jsonify(ok=False, err="no text"), 400
        wav = to_wav(src)
        t0 = time.time()
        out = {}
        def job_w():
            try: out["w"] = _whisper(wav, lang, hint, fast=True)
            except Exception as e: out["w_err"] = str(e)
        def job_p():
            try: out["p"] = _pron_eval(wav, text)
            except Exception as e: out["p_err"] = str(e)
        with _lock:
            job_w(); job_p()  # tuần tự — tránh deadlock CTranslate2+PyTorch chạy song song
        w = out.get("w") or {"text": "", "words": [], "duration": 0.0}
        p = out.get("p") or {"ok": False}
        resp = {"ok": True, "text": w["text"], "words": w["words"], "duration": w["duration"],
                "took": round(time.time() - t0, 2)}
        if p.get("ok"):
            resp.update(accuracy=p["accuracy"], target=p["target"], said=p["said"], phones=p["phones"])
        return jsonify(**resp)
    except subprocess.CalledProcessError:
        return jsonify(ok=False, err="audio decode failed"), 400
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 500
    finally:
        _cleanup([src, wav]);  os.rmdir(tmpd) if os.path.isdir(tmpd) else None


@app.post("/grade_ph")
def grade_ph():
    # Chấm 1 từ CHỈ bằng âm vị (không Whisper, không cửa sổ 30s) -> nhanh ~0.7s
    saved, err = _save_wav(request)
    if err: return err
    tmpd, src = saved
    current = (request.form.get("text", "") or "").strip()
    wav = None
    try:
        if not current:
            return jsonify(ok=False, err="no text"), 400
        wav = to_wav(src)
        t0 = time.time()
        with _lock:
            r = _grade_phonemes(wav, current)
        return jsonify(took=round(time.time() - t0, 2), **r)
    except subprocess.CalledProcessError:
        return jsonify(ok=False, err="audio decode failed"), 400
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 500
    finally:
        _cleanup([src, wav]);  os.rmdir(tmpd) if os.path.isdir(tmpd) else None


# ===== /score — trả điểm + màu + nhãn sẵn (mirror logic chấm samio) =====
import re as _re

def _dnorm(s):
    s = (s or "").lower()
    s = _re.sub(r"[^\w\s]", " ", s, flags=_re.UNICODE)
    return _re.sub(r"\s+", " ", s).strip()

def _wsim(a, b):
    a = a.strip(); b = b.strip()
    if not a or not b: return 0.0
    if a == b: return 1.0
    la, lb = len(a), len(b)
    d = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = d[0]; d[0] = i
        for j in range(1, lb + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = cur
    m = max(la, lb)
    return max(0.0, 1.0 - d[lb] / m)

def _align_words(ref, hyp):
    la, lb = len(ref), len(hyp)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1): d[i][0] = i
    for j in range(lb + 1): d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    st = ['miss'] * la; i, j = la, lb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]: st[i - 1] = 'ok'; i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1: st[i - 1] = 'sub'; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1: st[i - 1] = 'miss'; i -= 1
        else: j -= 1
    ok = sum(1 for x in st if x == 'ok')
    return st, ok

def _band(s):
    if s >= 8: return {"color": "#16a34a", "label": "Đạt", "emoji": "✅"}
    return {"color": "#dc2626", "label": "Không đạt", "emoji": "❌"}


@app.post("/score")
def score_ep():
    saved, err = _save_wav(request)
    if err: return err
    tmpd, src = saved
    text = (request.form.get("text", "") or "").strip()
    words_hint = (request.form.get("words", "") or "").strip()
    wav = None
    try:
        if not text:
            return jsonify(ok=False, err="no text"), 400
        wav = to_wav(src)
        t0 = time.time()
        tnorm = _dnorm(text)
        ntoks = len(tnorm.split()) if tnorm else 0
        hint = "" if ntoks > 1 else (words_hint if words_hint else text)
        with _lock:
            w = _whisper(wav, "en", hint, fast=True)
            p = _pron_eval(wav, text)
        transcript = (w.get("text") or "").strip()
        pacc = p.get("accuracy") if p.get("ok") else None
        pacc = None if pacc is None else pacc / 100.0
        phones = p.get("phones") if p.get("ok") else []
        marks = None
        if not transcript:
            score = 0.0; status = "miss"
        elif ntoks > 1:
            ref = tnorm.split(); hyp = _dnorm(transcript).split()
            st, ok = _align_words(ref, hyp)
            wr = ok / max(1, len(ref))
            bonus = wr * (1.0 if pacc is None else 3.0 * pacc)
            score = round(min(10.0, wr * 7.0 + bonus), 1)
            status = "ok" if score >= 8.5 else ("warn" if score >= 7 else "sub")
            marks = [{"w": ref[k], "ok": st[k] == "ok"} for k in range(len(ref))]
        else:
            best = 0.0
            for wd in _dnorm(transcript).split():
                best = max(best, _wsim(wd, tnorm))
            if tnorm and _dnorm(transcript) == tnorm: best = 1.0
            if best >= 0.85:
                bonus = 1.0 if pacc is None else 3.0 * pacc
                score = round(min(10.0, 7.0 + bonus), 1)
                status = "ok" if score >= 8.5 else "warn"
            elif best >= 0.5:
                score = round(4.0 + 2.0 * best, 1); status = "sub"
            else:
                score = round(3.0 * best, 1); status = "sub"
        return jsonify(ok=True, score=score, status=status, band=_band(score),
                       heard=transcript, marks=marks, phones=phones,
                       took=round(time.time() - t0, 2))
    except subprocess.CalledProcessError:
        return jsonify(ok=False, err="audio decode failed"), 400
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 500
    finally:
        _cleanup([src, wav])
        if os.path.isdir(tmpd): os.rmdir(tmpd)


# ===== /tts — Kokoro TTS (nạp model lười, không làm chậm cold-start chấm điểm) =====
# Giọng phổ biến: a*=American, b*=British; *f*=nữ, *m*=nam
_TTS_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
               "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
_kokoro = {}
_kokoro_lock = threading.Lock()

def _kokoro_pipe(lang):
    if lang not in _kokoro:
        with _kokoro_lock:
            if lang not in _kokoro:
                from kokoro import KPipeline
                _kokoro[lang] = KPipeline(lang_code=lang)
    return _kokoro[lang]


@app.post("/tts")
def tts():
    import re as _re2, base64 as _b64
    text = (request.form.get("text", "") or "").strip()[:2000]
    voice = _re2.sub(r"[^a-z_]", "", (request.form.get("voice", "") or "af_heart").lower()) or "af_heart"
    if voice not in _TTS_VOICES:
        voice = "af_heart"
    fmt = (request.form.get("format", "mp3") or "mp3").lower()
    try:
        speed = float(request.form.get("speed", "1") or "1")
    except Exception:
        speed = 1.0
    if not text:
        return jsonify(ok=False, err="no text"), 400
    lang = 'b' if voice[:1] == 'b' else 'a'   # bf_/bm_ -> British
    try:
        t0 = time.time()
        pipe = _kokoro_pipe(lang)
        chunks = []
        with _lock:  # 1 GPU — tuần tự như phần chấm
            for _g, _p, audio in pipe(text, voice=voice, speed=speed):
                a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                chunks.append(a)
        if not chunks:
            return jsonify(ok=False, err="empty audio"), 200
        wav = np.concatenate(chunks).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            wpath = os.path.join(d, "a.wav")
            sf.write(wpath, wav, 24000, subtype="PCM_16")
            if fmt == "wav":
                data = open(wpath, "rb").read(); out_fmt = "wav"
            else:
                mpath = os.path.join(d, "a.mp3")
                subprocess.run(["ffmpeg", "-y", "-i", wpath, "-b:a", "96k", mpath],
                               check=True, capture_output=True)
                data = open(mpath, "rb").read(); out_fmt = "mp3"
        return jsonify(ok=True, audio_b64=_b64.b64encode(data).decode(), format=out_fmt,
                       voice=voice, dur=round(len(wav) / 24000.0, 2), took=round(time.time() - t0, 2))
    except Exception as e:
        return jsonify(ok=False, err=str(e)), 500


if __name__ == "__main__":
    _bind = os.environ.get("ASR_BIND", "127.0.0.1")
    _crt = os.environ.get("ASR_TLS_CERT"); _key = os.environ.get("ASR_TLS_KEY")
    _ssl = (_crt, _key) if _crt and _key and os.path.exists(_crt) and os.path.exists(_key) else None
    app.run(host=_bind, port=PORT, threaded=True, ssl_context=_ssl)

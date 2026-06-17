#!/usr/bin/env python3
# Tạo test_input.json cho RunPod local test từ 1 file audio.
#   python make_test_input.py path/to/audio.wav "apple" [route]
import base64, json, sys

audio = sys.argv[1]
text = sys.argv[2] if len(sys.argv) > 2 else "apple"
route = sys.argv[3] if len(sys.argv) > 3 else "score"

with open(audio, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {"input": {"route": route, "text": text,
                     "filename": audio.split("/")[-1], "audio_b64": b64}}
with open("test_input.json", "w") as f:
    json.dump(payload, f)
print(f"wrote test_input.json  route={route}  text={text!r}  bytes_b64={len(b64)}")

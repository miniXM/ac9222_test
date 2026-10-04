
import base64, json, wave, struct, math, os, sys, uuid, traceback
import urllib.request, urllib.error

# --- 1s 16kHz mono reference tone ---
sr = 16000
n = int(sr * 1.0)
frames = bytearray()
for i in range(n):
    v = int(12000 * math.sin(2 * math.pi * 440.0 * i / sr))
    frames += struct.pack("<h", v)
wp = "/tmp/h3_ref.wav"
w = wave.open(wp, "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
w.writeframes(bytes(frames)); w.close()
print("WAV bytes:", os.path.getsize(wp), flush=True)

b64 = base64.b64encode(open(wp, "rb").read()).decode()
data_uri = "data:audio/wav;base64," + b64

boundary = "----h3b" + uuid.uuid4().hex

def field(name, value):
    return (("--%s\r\n" % boundary).encode()
            + ('Content-Disposition: form-data; name="%s"\r\n\r\n' % name).encode()
            + value.encode() + b"\r\n")

def filefield(name, path, filename, ct):
    data = open(path, "rb").read()
    return (("--%s\r\n" % boundary).encode()
            + ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n' % (name, filename)).encode()
            + ("Content-Type: %s\r\n\r\n" % ct).encode()
            + data + b"\r\n")

body = b""
body += field("prompt", "A red apple slowly rotating on a white table, soft studio lighting, cinematic")
body += filefield("input_reference", "/tmp/h3_ref.png", "h3_ref.png", "image/png")
body += field("audio_reference", json.dumps({"audio_url": data_uri}))
body += field("width", "256")
body += field("height", "256")
body += field("num_frames", "25")
body += field("num_inference_steps", "8")
body += field("seed", "42")
body += field("fps", "24")
body += ("--%s--\r\n" % boundary).encode()
print("body bytes:", len(body), flush=True)

req = urllib.request.Request("http://localhost:8000/v1/videos/sync", data=body,
      headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary})
try:
    r = urllib.request.urlopen(req, timeout=5400)
    out = r.read(); code = r.status; ct = r.headers.get("Content-Type")
except urllib.error.HTTPError as e:
    out = e.read(); code = e.code; ct = e.headers.get("Content-Type")
except Exception:
    traceback.print_exc()
    open("/tmp/h3_req_a4.txt", "w").write("CLIENT_EXCEPTION\n")
    sys.exit(1)

open("/tmp/h3_out_a4.bin", "wb").write(out)
open("/tmp/h3_req_a4.txt", "w").write("HTTP=%s CT=%s SZ=%s\n" % (code, ct, len(out)))
print("DONE", code, len(out), flush=True)

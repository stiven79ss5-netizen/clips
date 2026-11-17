import os
import subprocess
import numpy as np
from faster_whisper import WhisperModel

URL = os.environ["VIDEO_URL"]
N = int(os.environ.get("CLIPS", "5"))          # сколько клипов
LEN = int(os.environ.get("CLIP_LEN", "40"))    # длина клипа, сек
VERTICAL = os.environ.get("VERTICAL", "true") == "true"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

WORK, OUT = "work", "output"
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def run(cmd, **kw):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def loudness(path):
    """Громкость по секундам (читаем поток, мало памяти)."""
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", "8000", "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    vals = []
    while True:
        buf = p.stdout.read(8000 * 2)
        if len(buf) < 16000:
            break
        x = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        vals.append(float(np.sqrt(np.mean(x * x))))
    p.wait()
    return np.array(vals)


def pick_peaks(rms, n, gap):
    """Самые громкие/эмоциональные моменты (крики, смех, реакции)."""
    k = 10
    sm = np.convolve(rms, np.ones(k) / k, mode="same")
    score = sm / (np.median(sm) + 1e-6)
    score[:30] = 0  # пропускаем начало
    s = score.copy()
    peaks = []
    for _ in range(n):
        i = int(np.argmax(s))
        if s[i] <= 0:
            break
        peaks.append(i)
        s[max(0, i - gap): i + gap] = 0
    return sorted(peaks)


def fmt(t):
    ms = int((t - int(t)) * 1000)
    t = int(t)
    return f"{t // 3600:02}:{t % 3600 // 60:02}:{t % 60:02},{ms:03}"


def write_srt(segments, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            n += 1
            f.write(f"{n}\n{fmt(s.start)} --> {fmt(s.end)}\n{text}\n\n")
    return n


# 1. Скачиваем только звук (быстро) и ищем моменты
run(["yt-dlp", "-f", "ba/b", "-o", f"{WORK}/audio.%(ext)s", URL])
audio = [f for f in os.listdir(WORK) if f.startswith("audio.")][0]
rms = loudness(f"{WORK}/{audio}")
total = len(rms)
print("Длительность, сек:", total)

peaks = pick_peaks(rms, N, LEN)
print("Найденные моменты (сек):", peaks)

model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

# 2. Для каждого момента: качаем отрезок, субтитры, монтаж
for idx, peak in enumerate(peaks, 1):
    start = max(0, peak - int(LEN * 0.65))
    end = min(total, start + LEN)
    raw = f"{WORK}/raw_{idx}.mp4"

    run(["yt-dlp",
         "-f", "bv*[height<=1080]+ba/b[height<=1080]",
         "--download-sections", f"*{start}-{end}",
         "--force-keyframes-at-cuts",
         "--merge-output-format", "mp4",
         "-o", f"{WORK}/raw_{idx}.%(ext)s", URL])

    segs, _ = model.transcribe(raw, vad_filter=True)
    srt = f"clip_{idx}.srt"
    has_subs = write_srt(segs, f"{WORK}/{srt}") > 0

    size = 10 if VERTICAL else 16
    style = (f"FontName=DejaVu Sans,FontSize={size},Bold=1,"
             "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "Outline=2,Alignment=2,MarginV=50")
    sub = f",subtitles={srt}:force_style='{style}'" if has_subs else ""

    if VERTICAL:
        vf = ("[0:v]split[a][b];"
              "[a]scale=1080:1920:force_original_aspect_ratio=increase,"
              "crop=1080:1920,boxblur=20:5[bg];"
              "[b]scale=1080:-2[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2{sub}[v]")
    else:
        vf = f"[0:v]scale=1920:-2{sub}[v]"

    run(["ffmpeg", "-y", "-v", "error", "-i", f"raw_{idx}.mp4",
         "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-movflags", "+faststart",
         f"../{OUT}/clip_{idx}.mp4"], cwd=WORK)

print("Готово. Клипы в папке output/")
  

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI(title="Whisper Tiny API - Free Vercel")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model một lần duy nhất + dùng tiny để siêu nhẹ
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.get("/")
def home():
    return {"status": "ok", "model": "tiny.int8", "tip": "Dùng POST /transcribe để upload file"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("vi", description="vi, en, ja, zh..."),
    format: str = Query("json", description="json / text / srt")
):
    # Chỉ cho phép file nhỏ để tránh timeout (free chỉ 60s)
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm')):
        raise HTTPException(400, "Chỉ hỗ trợ mp3, wav, m4a, ogg, flac, webm")

    # Giới hạn kích thước file ~25MB (khoảng 20-25 phút audio là max an toàn cho free)
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(400, "File quá lớn! Maximum 25MB (khoảng 20 phút audio)")

    # Lưu tạm
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    temp_file.write(content)
    temp_file.close()

    try:
        segments, info = model.transcribe(
            temp_file.name,
            language=language if language != "auto" else None,
            beam_size=5,
            word_timestamps=False  # tắt để nhanh hơn
        )

        result_segments = []
        full_text = []

        for seg in segments:
            result_segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip()
            })
            full_text.append(seg.text.strip())

        os.unlink(temp_file.name)

        if format == "text":
            return PlainTextResponse(" ".join(full_text))
        elif format == "srt":
            srt = ""
            for i, s in enumerate(result_segments, 1):
                start = format_time(s["start"])
                end = format_time(s["end"])
                srt += f"{i}\n{start} --> {end}\n{s['text']}\n\n"
            return PlainTextResponse(srt, media_type="text/plain")
        else:
            return JSONResponse({
                "success": True,
                "language": info.language,
                "language_prob": round(info.language_probability, 2),
                "duration": round(info.duration, 2),
                "segments": result_segments,
                "text": " ".join(full_text)
            })

    except Exception as e:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(500, f"Lỗi xử lý: {str(e)}")

def format_time(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@app.get("/health")
def health():
    return {"status": "healthy", "model": "tiny.int8"}

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os
import json

app = FastAPI(
    title="Whisper Transcription API",
    description="Chuyển audio thành text với Faster-Whisper",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def get_model():
    global model
    if model is None:
        model = WhisperModel("base", device="cpu", compute_type="int8")
    return model

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Whisper Transcription API",
        "endpoints": {
            "/transcribe": "POST - Upload audio file",
            "/transcribe/url": "GET - Transcribe from URL"
        }
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query("vi", description="Mã ngôn ngữ (vi, en, etc)"),
    format: str = Query("json", description="Output format: json, text, srt")
):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file audio: mp3, wav, m4a, ogg, flac")
    
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"whisper_{file.filename}")
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        whisper_model = get_model()
        segments, info = whisper_model.transcribe(
            temp_path,
            language=language,
            beam_size=5
        )
        
        results = []
        full_text = []
        
        for segment in segments:
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())
        
        os.remove(temp_path)
        
        if format == "text":
            return PlainTextResponse(" ".join(full_text))
        
        elif format == "srt":
            srt_content = []
            for i, seg in enumerate(results, 1):
                start_time = format_timestamp(seg["start"])
                end_time = format_timestamp(seg["end"])
                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{seg['text']}\n")
            return PlainTextResponse("\n".join(srt_content))
        
        else:
            return JSONResponse({
                "success": True,
                "language": info.language,
                "duration": round(info.duration, 2),
                "segments": results,
                "full_text": " ".join(full_text)
            })
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

@app.get("/health")
def health():
    return {"status": "healthy"}

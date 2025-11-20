from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import tempfile
import os

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

# Global model variable
model = None

def get_model():
    """Lazy load model để tiết kiệm memory"""
    global model
    if model is None:
        try:
            from faster_whisper import WhisperModel
            # Sử dụng model nhỏ nhất để fit vào Vercel limits
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Không thể load model Whisper")
    return model

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Whisper Transcription API",
        "endpoints": {
            "/transcribe": "POST - Upload audio file",
            "/health": "GET - Health check"
        }
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query("vi", description="Mã ngôn ngữ (vi, en, etc)"),
    format: str = Query("json", description="Output format: json, text, srt")
):
    """
    Transcribe audio file
    - Hỗ trợ: mp3, wav, m4a, ogg, flac
    - Max file size: 50MB (Vercel limit)
    """
    # Kiểm tra extension
    allowed_extensions = ('.mp3', '.wav', '.m4a', '.ogg', '.flac')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Chỉ hỗ trợ file: {', '.join(allowed_extensions)}"
        )
    
    # Tạo temp file
    temp_path = None
    try:
        # Đọc file content
        content = await file.read()
        
        # Kiểm tra size (50MB limit)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File quá lớn. Giới hạn 50MB")
        
        # Tạo temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"whisper_{os.getpid()}_{file.filename}")
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Load model
        whisper_model = get_model()
        
        # Transcribe
        segments, info = whisper_model.transcribe(
            temp_path,
            language=language if language else None,
            beam_size=3,  # Giảm để nhanh hơn
            vad_filter=True,  # Lọc silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect results
        results = []
        full_text = []
        
        for segment in segments:
            results.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())
        
        # Format response
        if format == "text":
            return PlainTextResponse(" ".join(full_text))
        
        elif format == "srt":
            srt_content = []
            for i, seg in enumerate(results, 1):
                start_time = format_timestamp(seg["start"])
                end_time = format_timestamp(seg["end"])
                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{seg['text']}\n")
            return PlainTextResponse("\n".join(srt_content))
        
        else:  # json
            return JSONResponse({
                "success": True,
                "language": info.language,
                "duration": round(info.duration, 2),
                "segments": results,
                "full_text": " ".join(full_text)
            })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Mangum handler cho Vercel
handler = Mangum(app)

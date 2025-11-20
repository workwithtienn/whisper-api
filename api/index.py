import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from groq import Groq

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), format: str = "json"):
    allowed = ('.mp3', '.wav', '.m4a', '.ogg', '.flac')
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail="Only audio files: mp3, wav, m4a, ogg, flac")
    temp_dir = tempfile.gettempdir()
    temp_path = temp_dir + "/" + file.filename
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        with open(temp_path, "rb") as audio:
            result = client.audio.transcriptions.create(
                file=("audio.wav", audio.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json"
            )
        if isinstance(result, dict):
            text = result.get("text", "")
            segments = result.get("segments", [])
        else:
            text = getattr(result, "text", "") or ""
            segments = getattr(result, "segments", []) or []
        if format == "text":
            return PlainTextResponse(text)
        if format == "srt":
            srt_lines = []
            for i, seg in enumerate(segments, 1):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                seg_text = seg.get("text", "")
                hrs_s = int(start // 3600)
                mins_s = int((start % 3600) // 60)
                secs_s = int(start % 60)
                ms_s = int((start % 1) * 1000)
                hrs_e = int(end // 3600)
                mins_e = int((end % 3600) // 60)
                secs_e = int(end % 60)
                ms_e = int((end % 1) * 1000)
                start_ts = f"{hrs_s:02}:{mins_s:02}:{secs_s:02},{ms_s:03}"
                end_ts = f"{hrs_e:02}:{mins_e:02}:{secs_e:02},{ms_e:03}"
                srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{seg_text}\n")
            return PlainTextResponse("\n".join(srt_lines))
        return JSONResponse({"success": True, "text": text, "segments": segments})
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

@app.get("/")
def root():
    return {"status": "ok", "message": "Groq Whisper API"}

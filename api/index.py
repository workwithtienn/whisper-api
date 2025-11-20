from fastapi import FastAPI, UploadFile, File
from groq import Groq
import uvicorn

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio = await file.read()
    response = client.audio.transcriptions.create(
        file=("audio.wav", audio),
        model="whisper-large-v3-turbo"
    )
    return {"text": response["text"]}

def handler(event, context):
    return uvicorn.run(app)

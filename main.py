from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pydub import AudioSegment
import soundfile as sf
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path + "_conv.wav"
    audio.export(wav_path, format="wav")
    data, sr = sf.read(wav_path)
    return data, sr

@app.post("/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    # Save uploaded file
    input_path = f"input_{uuid4()}_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Load audio (converts ANY format to WAV)
    audio, sr = load_audio(input_path)

    # TODO: your processing here
    # For now we send back original
    output = io.BytesIO()
    sf.write(output, audio, sr, format="WAV")
    output.seek(0)

    # Cleanup
    os.remove(input_path)

    return Response(
        content=output.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=enhanced.wav"}
    )

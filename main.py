from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uuid
import os
from processing import process_audio_file

app = FastAPI()

@app.post("/enhance")
async def enhance_audio(file: UploadFile = File(...)):
    input_path = f"input_{uuid.uuid4()}.wav"
    output_path = f"output_{uuid.uuid4()}.wav"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    process_audio_file(input_path, output_path)

    return FileResponse(output_path, media_type="audio/wav")

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import uvicorn
import whisper
import json
import tempfile
import os
from pathlib import Path

app = FastAPI()

VERSION = "0.1.0"
model = None


@app.on_event("startup")
async def startup_event():
    global model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Whisper model loaded successfully")


@app.get("/")
async def root():
    return JSONResponse({"version": VERSION})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Receive audio file data
        audio_data = await websocket.receive_bytes()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            # Transcribe with verbose mode to get segments
            result = model.transcribe(
                temp_audio_path,
                verbose=False,
                task="transcribe"
            )
            
            # Emit each segment as it's available
            for segment in result["segments"]:
                subtitle_event = {
                    "type": "subtitle",
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                }
                await websocket.send_text(json.dumps(subtitle_event))
            
            # Send completion event
            completion_event = {
                "type": "completed",
                "full_text": result["text"].strip()
            }
            await websocket.send_text(json.dumps(completion_event))
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except Exception as e:
        error_event = {
            "type": "error",
            "message": str(e)
        }
        await websocket.send_text(json.dumps(error_event))
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

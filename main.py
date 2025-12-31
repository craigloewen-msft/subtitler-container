from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
import uvicorn
from faster_whisper import WhisperModel
import json
import tempfile
import os
import asyncio
from pathlib import Path

app = FastAPI()

VERSION = "0.1.0"
model = None


@app.on_event("startup")
async def startup_event():
    global model
    print("Loading Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Whisper model loaded successfully")


@app.get("/")
async def root():
    return JSONResponse({"version": VERSION})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    temp_audio_path = None
    
    try:
        # Receive audio file data
        audio_data = await websocket.receive_bytes()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        # Transcribe with streaming - segments are yielded as they're decoded
        segments, info = model.transcribe(
            temp_audio_path,
            task="transcribe"
        )
        
        full_text_parts = []
        segment_count = 0
        # Emit each segment as it's available (true streaming)
        for segment in segments:
            segment_count += 1
            subtitle_event = {
                "type": "subtitle",
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            }
            await websocket.send_text(json.dumps(subtitle_event))
            full_text_parts.append(segment.text)
            # Yield control to event loop to keep connection alive
            await asyncio.sleep(0)
        
        print(f"Processed {segment_count} segments")
        
        # Send completion event
        completion_event = {
            "type": "completed",
            "full_text": " ".join(full_text_parts).strip()
        }
        await websocket.send_text(json.dumps(completion_event))
            
    except WebSocketDisconnect:
        # Client disconnected - just clean up, don't try to send anything
        pass
    except Exception as e:
        # Try to send error if connection is still open
        try:
            error_event = {
                "type": "error",
                "message": str(e)
            }
            await websocket.send_text(json.dumps(error_event))
        except (WebSocketDisconnect, RuntimeError):
            # Connection already closed, can't send error
            pass
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        
        # Try to close the websocket if it's still open
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            # Already closed
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

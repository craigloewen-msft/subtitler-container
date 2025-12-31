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
import base64

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


async def transcribe_from_position(websocket: WebSocket, audio_path: str, seek_time: float, cancel_event: asyncio.Event):
    """
    Transcribe audio from a specific position, cancellable via cancel_event.
    """
    try:
        # Send processing started event
        await websocket.send_text(json.dumps({
            "type": "processing_started",
            "seek_time": seek_time
        }))
        
        # Transcribe with streaming
        segments, info = model.transcribe(
            audio_path,
            task="transcribe"
        )
        
        full_text_parts = []
        segment_count = 0
        
        # Process segments
        for segment in segments:
            # Check if we should cancel
            if cancel_event.is_set():
                await websocket.send_text(json.dumps({
                    "type": "processing_cancelled",
                    "seek_time": seek_time
                }))
                return
            
            # Skip segments that end before the seek time
            if segment.end < seek_time:
                continue
            
            # For segments that overlap the seek time, adjust the start time
            start_time = max(segment.start, seek_time)
            
            segment_count += 1
            subtitle_event = {
                "type": "subtitle",
                "start": start_time,
                "end": segment.end,
                "text": segment.text.strip()
            }
            await websocket.send_text(json.dumps(subtitle_event))
            full_text_parts.append(segment.text)
            
            # Yield control to event loop
            await asyncio.sleep(0)
        
        # Only send completion if not cancelled
        if not cancel_event.is_set():
            print(f"Processed {segment_count} segments (seek_time: {seek_time}s)")
            completion_event = {
                "type": "completed",
                "full_text": " ".join(full_text_parts).strip(),
                "seek_time": seek_time
            }
            await websocket.send_text(json.dumps(completion_event))
    except Exception as e:
        if not cancel_event.is_set():
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    temp_audio_path = None
    transcription_task = None
    cancel_event = asyncio.Event()
    
    try:
        while True:
            # Receive command
            message_text = await websocket.receive_text()
            command = json.loads(message_text)
            command_type = command.get("type")
            
            if command_type == "load":
                # Load audio data
                audio_base64 = command.get("audio_data")
                if not audio_base64:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "audio_data is required for load command"
                    }))
                    continue
                
                # Cancel any existing transcription
                if transcription_task and not transcription_task.done():
                    cancel_event.set()
                    await transcription_task
                
                # Clean up old temp file if it exists
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                
                # Decode and save audio
                audio_data = base64.b64decode(audio_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(audio_data)
                    temp_audio_path = temp_audio.name
                
                await websocket.send_text(json.dumps({
                    "type": "loaded",
                    "message": "Audio loaded successfully"
                }))
                
                # Start transcription from beginning
                cancel_event = asyncio.Event()
                transcription_task = asyncio.create_task(
                    transcribe_from_position(websocket, temp_audio_path, 0.0, cancel_event)
                )
            
            elif command_type == "seek":
                # Seek to a specific position
                seek_time = command.get("seek_time", 0.0)
                
                if not temp_audio_path:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No audio loaded. Use 'load' command first."
                    }))
                    continue
                
                # Cancel current transcription
                if transcription_task and not transcription_task.done():
                    cancel_event.set()
                    await transcription_task
                
                # Start new transcription from seek position
                cancel_event = asyncio.Event()
                transcription_task = asyncio.create_task(
                    transcribe_from_position(websocket, temp_audio_path, seek_time, cancel_event)
                )
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown command type: {command_type}"
                }))
            
    except WebSocketDisconnect:
        # Client disconnected - just clean up
        pass
    except Exception as e:
        # Try to send error if connection is still open
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except (WebSocketDisconnect, RuntimeError):
            pass
    finally:
        # Cancel any running transcription
        if transcription_task and not transcription_task.done():
            cancel_event.set()
            try:
                await transcription_task
            except:
                pass
        
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        
        # Try to close the websocket if it's still open
        try:
            await websocket.close()
        except (WebSocketDisconnect, RuntimeError):
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

app = FastAPI()

VERSION = "0.1.0"
model = None


@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    logger.info("Whisper model loaded successfully")


@app.get("/")
async def root():
    return JSONResponse({"version": VERSION})


def find_overlapping_translation(orig_start: float, orig_end: float, translation_segments: list) -> str:
    """
    Find translation segments that overlap with the original segment's time range.
    Combines text from all overlapping segments, weighted by overlap amount.
    """
    overlapping_texts = []
    
    for trans_seg in translation_segments:
        # Calculate overlap
        overlap_start = max(orig_start, trans_seg.start)
        overlap_end = min(orig_end, trans_seg.end)
        overlap_duration = overlap_end - overlap_start
        
        # If there's meaningful overlap (more than 10% of original segment)
        orig_duration = orig_end - orig_start
        if overlap_duration > 0 and overlap_duration / orig_duration > 0.1:
            overlapping_texts.append(trans_seg.text.strip())
    
    # Combine all overlapping translations
    if overlapping_texts:
        return " ".join(overlapping_texts)
    
    # Fallback: find the closest segment by midpoint
    orig_mid = (orig_start + orig_end) / 2
    closest_seg = min(translation_segments, 
                     key=lambda seg: abs((seg.start + seg.end) / 2 - orig_mid))
    return closest_seg.text.strip()


async def transcribe_from_position(websocket: WebSocket, audio_path: str, seek_time: float, cancel_event: asyncio.Event):
    """
    Transcribe audio from a specific position, cancellable via cancel_event.
    Detects language and provides both original and English translation, streaming results.
    Processes both streams simultaneously and sends events as soon as matches are ready.
    """
    try:
        # Send processing started event
        await websocket.send_text(json.dumps({
            "type": "processing_started",
            "seek_time": seek_time
        }))
        
        # First, do a quick language detection
        logger.info("Detecting language...")
        
        def detect_language():
            segments_iter, info = model.transcribe(audio_path, task="transcribe")
            return info.language, info.language_probability
        
        detected_language, language_probability = await asyncio.to_thread(detect_language)
        logger.info(f"Detected language: {detected_language} (probability: {language_probability:.2f})")
        
        # Now start both transcription and translation streams
        original_queue = asyncio.Queue()
        translation_queue = asyncio.Queue()
        
        # Capture the event loop
        loop = asyncio.get_event_loop()
        
        def produce_segments(task_type: str, queue: asyncio.Queue, loop):
            """Run transcription in thread and put segments into queue"""
            try:
                segments_iter, _ = model.transcribe(audio_path, task=task_type)
                for segment in segments_iter:
                    asyncio.run_coroutine_threadsafe(queue.put(segment), loop).result()
            except Exception as e:
                logger.error(f"Error in {task_type}: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
        
        # Start both transcriptions in parallel threads
        import concurrent.futures
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        executor.submit(produce_segments, "transcribe", original_queue, loop)
        
        if detected_language != "en":
            executor.submit(produce_segments, "translate", translation_queue, loop)
        
        # Track segments for matching
        translation_buffer = []  # Buffer of all translation segments seen so far
        full_text_parts = []
        full_text_en_parts = []
        segment_count = 0
        
        # Process original segments as they arrive
        while True:
            if cancel_event.is_set():
                await websocket.send_text(json.dumps({
                    "type": "processing_cancelled",
                    "seek_time": seek_time
                }))
                return
            
            # Get next original segment (with timeout to check cancellation)
            try:
                orig_segment = await asyncio.wait_for(original_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            
            if orig_segment is None:  # Stream ended
                break
            
            # Skip segments before seek time
            if orig_segment.end < seek_time:
                continue
            
            # For non-English, wait until we have translation coverage
            if detected_language != "en":
                # Collect translation segments until we cover this original segment
                while True:
                    # Check if we already have coverage
                    if translation_buffer and translation_buffer[-1].end >= orig_segment.end:
                        break
                    
                    # Get more translation segments
                    try:
                        trans_segment = await asyncio.wait_for(translation_queue.get(), timeout=0.1)
                        if trans_segment is None:  # Translation stream ended
                            break
                        translation_buffer.append(trans_segment)
                    except asyncio.TimeoutError:
                        # Check if we have enough coverage anyway
                        if translation_buffer and translation_buffer[-1].end >= orig_segment.end:
                            break
                        continue
            
            # Now we can send this segment
            start_time = max(orig_segment.start, seek_time)
            segment_count += 1
            
            # Find matching translation by time overlap
            if detected_language != "en" and translation_buffer:
                text_en = find_overlapping_translation(
                    orig_segment.start, 
                    orig_segment.end, 
                    translation_buffer
                )
            else:
                text_en = orig_segment.text.strip()
            
            subtitle_event = {
                "type": "subtitle",
                "start": start_time,
                "end": orig_segment.end,
                "text": orig_segment.text.strip(),
                "text_en": text_en,
                "language": detected_language
            }
            await websocket.send_text(json.dumps(subtitle_event))
            full_text_parts.append(orig_segment.text)
            full_text_en_parts.append(text_en)
            
            # Yield control to event loop
            await asyncio.sleep(0)
        
        # Clean up executor
        executor.shutdown(wait=False)
        
        # Only send completion if not cancelled
        if not cancel_event.is_set():
            logger.info(f"Processed {segment_count} segments (seek_time: {seek_time}s, language: {detected_language})")
            completion_event = {
                "type": "completed",
                "full_text": " ".join(full_text_parts).strip(),
                "full_text_en": " ".join(full_text_en_parts).strip(),
                "language": detected_language,
                "seek_time": seek_time
            }
            await websocket.send_text(json.dumps(completion_event))
    except Exception as e:
        logger.exception(f"ERROR in transcribe_from_position: {e}")
        if not cancel_event.is_set():
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except Exception as send_error:
                logger.error(f"Failed to send error message: {send_error}")


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

import asyncio
import websockets
import json
import subprocess
import time
import sys
import requests
from pathlib import Path
from threading import Thread

# Color codes
COLOR_RESET = "\033[0m"
COLOR_SERVER = "\033[94m"  # Blue
COLOR_TEST = "\033[96m"    # Cyan
COLOR_SUCCESS = "\033[92m" # Green
COLOR_WARNING = "\033[93m" # Yellow
COLOR_ERROR = "\033[91m"   # Red


def stream_output(pipe, prefix, color):
    """Stream output from subprocess with colored prefix"""
    for line in iter(pipe.readline, b''):
        if line:
            print(f"{color}{prefix}{line.decode().rstrip()}{COLOR_RESET}")
    pipe.close()


def wait_for_server(timeout=60):
    """Wait for the server to be ready"""
    print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Waiting for server to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print(f"{COLOR_TEST}.{COLOR_RESET}", end="", flush=True)
    
    print(f"\n{COLOR_TEST}[TEST]{COLOR_RESET} Timeout waiting for server")
    return False


async def test_websocket():
    # Read the audio file
    audio_path = Path("test.mp3")
    if not audio_path.exists():
        print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Error: {audio_path} not found!")
        return
    
    print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Loading audio file: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    
    print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Audio file size: {len(audio_data)} bytes")
    
    # Connect to WebSocket
    uri = "ws://localhost:8000/ws"
    print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Connected! Sending audio data...")
        
        # Send audio data as bytes
        await websocket.send(audio_data)
        print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Audio data sent. Waiting for transcription...\n")
        
        print(COLOR_SUCCESS + "="*60)
        print("TRANSCRIPTION RESULTS")
        print("="*60 + COLOR_RESET + "\n")
        
        # Receive and print subtitle events
        while True:
            try:
                message = await websocket.recv()
                event = json.loads(message)
                
                if event["type"] == "subtitle":
                    print(f"{COLOR_WARNING}[{event['start']:6.2f}s - {event['end']:6.2f}s]{COLOR_RESET} {event['text']}")
                elif event["type"] == "completed":
                    print(f"\n{COLOR_SUCCESS}" + "="*60)
                    print("TRANSCRIPTION COMPLETE")
                    print("="*60 + COLOR_RESET)
                    print(f"\n{COLOR_SUCCESS}Full text:{COLOR_RESET} {event['full_text']}\n")
                    break
                elif event["type"] == "error":
                    print(f"{COLOR_ERROR}[TEST] Error: {event['message']}{COLOR_RESET}")
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Connection closed")
                break


def main():
    print(COLOR_SUCCESS + "="*60)
    print("WHISPER WEBSOCKET TRANSCRIPTION TEST")
    print("="*60 + COLOR_RESET + "\n")
    
    # Start the server
    print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Starting server...")
    server_process = subprocess.Popen(
        ["uv", "run", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Start threads to stream server output
    stdout_thread = Thread(
        target=stream_output, 
        args=(server_process.stdout, "[SERVER] ", COLOR_SERVER),
        daemon=True
    )
    stderr_thread = Thread(
        target=stream_output, 
        args=(server_process.stderr, "[SERVER] ", COLOR_SERVER),
        daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for server to be ready
    if not wait_for_server():
        print(f"{COLOR_ERROR}[TEST]{COLOR_RESET} Server failed to start")
        server_process.terminate()
        return
    
    print()  # Add blank line for readability
    
    try:
        # Run the WebSocket test
        asyncio.run(test_websocket())
    finally:
        # Stop the server
        print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Stopping server...")
        server_process.terminate()
        server_process.wait()
        print(f"{COLOR_TEST}[TEST]{COLOR_RESET} Server stopped\n")


if __name__ == "__main__":
    main()


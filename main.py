import os
import tempfile
import math
from typing import List, Dict

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Import Pyannote VAD model components
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

# --- FastAPI App and Template Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- VAD Model Initialization ---
# This happens once when the server starts.
# It might take a moment to download models on first run.
MODELS = {}

print("Loading VAD models...")
try:
    # Load Silero Model
    silero_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    MODELS['silero'] = (silero_model, utils)
    print("✅ Silero VAD model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading Silero VAD model: {e}")

try:
    # Load Pyannote Model
    # Note: For some versions/environments, you might need a Hugging Face auth token.
    # use_auth_token=os.environ.get("HUGGING_FACE_TOKEN")
    pyannote_model = Model.from_pretrained(
        "pyannote/segmentation",
        use_auth_token=False 
    )
    MODELS['pyannote'] = pyannote_model
    print("✅ Pyannote VAD model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading Pyannote VAD model: {e}")

# --- VAD Processing Functions ---
def apply_vad_silero(audio_path: str) -> List[Dict[str, float]]:
    """Applies Silero VAD and returns speech segments."""
    model, (get_speech_timestamps, _, read_audio, *_) = MODELS['silero']
    SAMPLING_RATE = 16000
    wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
    
    speech_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=SAMPLING_RATE,
        threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100
    )
    
    return [
        {"start": round(ts['start'] / SAMPLING_RATE, 3), "end": round(ts['end'] / SAMPLING_RATE, 3)}
        for ts in speech_timestamps
    ]

def apply_vad_pyannote(audio_path: str) -> List[Dict[str, float]]:
    """Applies Pyannote VAD and returns speech segments."""
    pipeline = VoiceActivityDetection(segmentation=MODELS['pyannote'])
    pipeline.instantiate({
        "min_duration_on": 0.25, "min_duration_off": 0.1,
        "onset": 0.5, "offset": 0.5
    })
    vad_result = pipeline(audio_path)
    
    return [
        {"start": round(segment.start, 3), "end": round(segment.end, 3)}
        for segment in vad_result.get_timeline()
    ]

VAD_FUNCTIONS = {
    "silero": apply_vad_silero,
    "pyannote": apply_vad_pyannote
}

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    """Serves the main HTML webpage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    vad_model: str = Form("silero"),
    split_size: int = Form(0)
):
    """
    Accepts a .wav file and analysis options, performs VAD, and returns segments.
    Implements chunking logic for large files based on `split_size`.
    """
    if vad_model not in VAD_FUNCTIONS:
        raise HTTPException(status_code=400, detail="Invalid VAD model selected.")
    
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .wav file.")

    main_tmp_path = ""
    try:
        # Save the uploaded file to a main temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            main_tmp_path = tmp_file.name

        # Load audio info to get duration
        waveform, sample_rate = torchaudio.load(main_tmp_path)
        duration = waveform.shape[1] / sample_rate
        
        # --- Chunking Logic ---
        if split_size <= 0 or split_size >= duration:
            # No splitting needed, process the whole file
            print(f"Processing entire file ({duration:.2f}s) with {vad_model} model...")
            segments = VAD_FUNCTIONS[vad_model](main_tmp_path)
            return JSONResponse(content=segments)
        else:
            # Splitting is needed
            print(f"Splitting file into {split_size}s chunks for {vad_model} model...")
            all_segments = []
            num_chunks = math.ceil(duration / split_size)

            for i in range(num_chunks):
                chunk_start_time = i * split_size
                chunk_end_time = min((i + 1) * split_size, duration)

                # Extract chunk waveform
                start_sample = int(chunk_start_time * sample_rate)
                end_sample = int(chunk_end_time * sample_rate)
                chunk_waveform = waveform[:, start_sample:end_sample]

                # Save chunk to its own temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_tmp_file:
                    chunk_path = chunk_tmp_file.name
                    torchaudio.save(chunk_path, chunk_waveform, sample_rate)
                
                # Run VAD on the chunk
                chunk_segments = VAD_FUNCTIONS[vad_model](chunk_path)
                
                # Adjust timestamps to be relative to the original audio
                for seg in chunk_segments:
                    all_segments.append({
                        "start": seg['start'] + chunk_start_time,
                        "end": seg['end'] + chunk_start_time
                    })
                
                # Clean up chunk file
                os.unlink(chunk_path)
            
            return JSONResponse(content=all_segments)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # Clean up the main temporary file
        if main_tmp_path and os.path.exists(main_tmp_path):
            os.unlink(main_tmp_path)
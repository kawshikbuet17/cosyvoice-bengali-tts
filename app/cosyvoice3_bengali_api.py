#!/usr/bin/env python3
"""CosyVoice3 Bengali TTS API.

This server is intentionally small and mirrors the working CLI path:

    CUDA_VISIBLE_DEVICES=1 python app/cosyvoice3_bengali_api.py \
      --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch200 \
      --mode instruct2 \
      --prompt-wav ./asset/kawshik_prompt.wav \
      --port 6008

Endpoints:
  - POST /tts/bytes : input {"text": "..."} and return raw PCM16LE bytes
  - POST /tts/wav   : input {"text": "..."} and return raw audio/wav bytes
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "bengali" / "cosyvoice3_emotion" / "local"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from emotion_tags import (  # noqa: E402
    DEFAULT_BASE_INSTRUCTION,
    DEFAULT_NEUTRAL_INSTRUCTION,
    convert_tagged_text,
    ensure_cosyvoice3_prefix,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("cosyvoice3_bengali_api")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Input Bengali text. Leading tags like [smile] are supported.")


RUNTIME = {
    "config": None,
    "cosyvoice": None,
    "prompt_wav": None,
    "prompt_info": None,
    "prompt_duration_seconds": None,
    "temp_prompt_path": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a minimal Bengali emotion-tagged CosyVoice3 REST API.")
    parser.add_argument("--host", default="0.0.0.0", help="API bind host")
    parser.add_argument("--port", type=int, default=6008, help="API bind port")
    parser.add_argument("--model-dir", required=True, help="CosyVoice3 model directory")
    parser.add_argument("--mode", choices=["instruct2", "cross_lingual"], default="instruct2",
                        help="Inference mode to use for all requests")
    parser.add_argument("--prompt-wav", required=True, help="Prompt WAV path used for all requests")
    parser.add_argument("--base-instruct", default=DEFAULT_BASE_INSTRUCTION,
                        help="Base instruction used when converting leading tags")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--text-frontend", action="store_true",
                        help="Enable CosyVoice text frontend. Default is off for Bengali Unicode safety.")
    parser.add_argument("--fp16", action="store_true", help="Load model in fp16")
    parser.add_argument("--load-trt", action="store_true", help="Load TensorRT engines if available")
    parser.add_argument("--load-vllm", action="store_true", help="Load vLLM backend if available")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).resolve()


def uses_cosyvoice3(model_dir: Path) -> bool:
    return "cosyvoice3" in str(model_dir).lower()


def prepare_inputs(raw_text: str, base_instruct: str, mode: str):
    converted = convert_tagged_text(raw_text, base_instruction=base_instruct)
    clean_text = converted["text"] or (raw_text or "").strip()

    if mode == "cross_lingual":
        if converted["tags"]:
            tts_text = converted["instruction"] + clean_text
        else:
            tts_text = ensure_cosyvoice3_prefix(clean_text)
        return tts_text, "", converted

    if converted["tags"]:
        return clean_text, converted["instruction"], converted

    return clean_text, ensure_cosyvoice3_prefix(DEFAULT_NEUTRAL_INSTRUCTION), converted


def prepare_prompt_audio_once(prompt_wav: Path, prompt_sr: int = 16000):
    """Preprocess the fixed prompt WAV once at startup."""
    info = torchaudio.info(str(prompt_wav))
    duration_seconds = 0.0
    if info.sample_rate > 0:
        duration_seconds = info.num_frames / float(info.sample_rate)

    waveform, sr = torchaudio.load(str(prompt_wav))
    if sr != prompt_sr:
        logger.info("Resampling prompt audio from %s Hz to %s Hz", sr, prompt_sr)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=prompt_sr)
        waveform = resampler(waveform)
        sr = prompt_sr

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    temp_path = Path(temp_file.name)
    torchaudio.save(str(temp_path), waveform, sr)
    prepared_info = torchaudio.info(str(temp_path))

    return temp_path, prepared_info, duration_seconds


def generate_speech(text: str, instruct_text: str):
    cosyvoice = RUNTIME["cosyvoice"]
    config = RUNTIME["config"]
    prompt_wav = RUNTIME["prompt_wav"]
    last_speech = None

    if config.mode == "cross_lingual":
        results = cosyvoice.inference_cross_lingual(
            text,
            str(prompt_wav),
            stream=False,
            speed=config.speed,
            text_frontend=config.text_frontend,
        )
    else:
        results = cosyvoice.inference_instruct2(
            text,
            instruct_text,
            str(prompt_wav),
            stream=False,
            speed=config.speed,
            text_frontend=config.text_frontend,
        )

    for item in results:
        last_speech = item["tts_speech"]

    return last_speech


def speech_to_pcm16le_bytes(speech_tensor) -> bytes:
    audio = speech_tensor.detach().cpu().squeeze(0).numpy()
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).round().astype("<i2")
    return pcm16.tobytes()


def speech_to_wav_bytes(speech_tensor, sample_rate: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torchaudio.save(str(tmp_path), speech_tensor, sample_rate)
        return tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to delete temporary WAV %s: %s", tmp_path, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = RUNTIME["config"]
    if config is None:
        raise RuntimeError("Server config was not initialized before startup.")

    model_dir = resolve_path(config.model_dir)
    prompt_wav = resolve_path(config.prompt_wav)

    if not model_dir.exists():
        raise RuntimeError(f"Model directory does not exist: {model_dir}")
    if not prompt_wav.exists():
        raise RuntimeError(f"Prompt WAV does not exist: {prompt_wav}")
    if not uses_cosyvoice3(model_dir):
        raise RuntimeError("This API expects a CosyVoice3 model directory.")

    logger.info("Loading model from %s", model_dir)
    RUNTIME["cosyvoice"] = AutoModel(
        model_dir=str(model_dir),
        load_trt=config.load_trt,
        load_vllm=config.load_vllm,
        fp16=config.fp16,
    )

    prepared_prompt, prompt_info, prompt_duration = prepare_prompt_audio_once(prompt_wav)
    if prompt_info.sample_rate < 16000:
        raise RuntimeError("Prompt audio sample rate is lower than 16 kHz.")
    if prompt_duration < 1.0:
        raise RuntimeError("Prompt audio is too short. Use at least 1 second.")

    RUNTIME["prompt_wav"] = prepared_prompt
    RUNTIME["prompt_info"] = prompt_info
    RUNTIME["prompt_duration_seconds"] = prompt_duration
    RUNTIME["temp_prompt_path"] = prepared_prompt

    logger.info("Prepared prompt WAV: %s", prepared_prompt)
    logger.info("API ready on %s:%s using mode=%s", config.host, config.port, config.mode)
    yield

    temp_prompt = RUNTIME.get("temp_prompt_path")
    if temp_prompt and Path(temp_prompt).exists():
        try:
            Path(temp_prompt).unlink()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to delete temporary prompt WAV %s: %s", temp_prompt, exc)


app = FastAPI(
    title="CosyVoice3 Bengali Emotion REST API",
    description="Minimal REST API for Bengali emotion-tagged CosyVoice3 inference.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/tts/bytes")
async def tts_bytes(request: TTSRequest):
    start_time = time.time()
    config = RUNTIME["config"]
    cosyvoice = RUNTIME["cosyvoice"]

    tts_text, instruct_text, converted = prepare_inputs(
        request.text,
        config.base_instruct,
        config.mode,
    )

    if converted["unsupported_tags"]:
        logger.warning("Unsupported tags ignored: %s", ", ".join(converted["unsupported_tags"]))

    speech = generate_speech(tts_text, instruct_text)
    if speech is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    audio_bytes = speech_to_pcm16le_bytes(speech)
    duration = len(speech[0]) / cosyvoice.sample_rate
    elapsed = time.time() - start_time

    headers = {
        "X-Mode": config.mode,
        "X-Sample-Rate": str(cosyvoice.sample_rate),
        "X-Channels": "1",
        "X-Encoding": "pcm_s16le",
        "X-Duration-Seconds": f"{duration:.6f}",
        "X-Processing-Time-Seconds": f"{elapsed:.6f}",
        "X-Converted-Instruction": converted["instruction"] if config.mode == "instruct2" else "",
        "X-Unsupported-Tags": ",".join(converted["unsupported_tags"]),
    }
    return Response(content=audio_bytes, media_type="application/octet-stream", headers=headers)


@app.post("/tts/wav")
async def tts_wav(request: TTSRequest):
    config = RUNTIME["config"]
    cosyvoice = RUNTIME["cosyvoice"]

    tts_text, instruct_text, converted = prepare_inputs(
        request.text,
        config.base_instruct,
        config.mode,
    )

    if converted["unsupported_tags"]:
        logger.warning("Unsupported tags ignored: %s", ", ".join(converted["unsupported_tags"]))

    speech = generate_speech(tts_text, instruct_text)
    if speech is None:
        raise HTTPException(status_code=500, detail="No audio generated")

    audio_bytes = speech_to_wav_bytes(speech, cosyvoice.sample_rate)
    headers = {
        "X-Mode": config.mode,
        "X-Converted-Instruction": converted["instruction"] if config.mode == "instruct2" else "",
        "X-Unsupported-Tags": ",".join(converted["unsupported_tags"]),
    }
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


def main():
    args = parse_args()
    RUNTIME["config"] = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

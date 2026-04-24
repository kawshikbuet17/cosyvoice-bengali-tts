#!/usr/bin/env python3
"""Bengali inference helper for CosyVoice3.

Run from examples/bengali/cosyvoice3, or from anywhere after installing the
CosyVoice environment. This script is intentionally small and uses the repo's
existing AutoModel inference API.
"""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
import torchaudio  # noqa: E402


DEFAULT_TEXT = "তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।"
COSYVOICE3_SYSTEM_PROMPT = "You are a helpful assistant.<|endofprompt|>"
DEFAULT_INSTRUCT = "You are a helpful assistant. Please speak in Bengali.<|endofprompt|>"


def uses_cosyvoice3(model_dir):
    return "cosyvoice3" in str(model_dir).lower()


def ensure_cosyvoice3_prefix(text):
    text = text.strip()
    if "<|endofprompt|>" in text:
        return text
    return COSYVOICE3_SYSTEM_PROMPT + text


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bengali inference with a CosyVoice3 model directory.")
    parser.add_argument("--model-dir", default="../../../pretrained_models/Fun-CosyVoice3-0.5B",
                        help="CosyVoice3 model directory containing cosyvoice3.yaml, llm.pt, flow.pt, hift.pt, and ONNX files")
    parser.add_argument("--mode", choices=["zero_shot", "cross_lingual", "instruct2"], default="zero_shot",
                        help="CosyVoice3 inference mode")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Bengali text to synthesize")
    parser.add_argument("--prompt-wav", required=True, help="Prompt audio path used for speaker/style reference")
    parser.add_argument("--prompt-text", default="",
                        help="Transcript of prompt audio. Strongly recommended for zero_shot mode")
    parser.add_argument("--instruct-text", default=DEFAULT_INSTRUCT,
                        help="Instruction text used by instruct2 mode")
    parser.add_argument("--output-dir", default="outputs/bengali_inference", help="Directory for generated wav files")
    parser.add_argument("--output-prefix", default="bengali", help="Generated wav filename prefix")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--stream", action="store_true", help="Use streaming inference")
    parser.add_argument("--text-frontend", action="store_true",
                        help="Enable CosyVoice text frontend. Default is off to preserve Bengali Unicode text")
    parser.add_argument("--fp16", action="store_true", help="Load model in fp16 when CUDA supports it")
    parser.add_argument("--load-trt", action="store_true", help="Load TensorRT engines if available")
    parser.add_argument("--load-vllm", action="store_true", help="Load vLLM backend if available")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def main():
    args = parse_args()
    model_dir = resolve_path(args.model_dir)
    prompt_wav = resolve_path(args.prompt_wav)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise SystemExit("model directory does not exist: {}".format(model_dir))
    if not prompt_wav.exists():
        raise SystemExit("prompt wav does not exist: {}".format(prompt_wav))
    if args.mode == "zero_shot" and not args.prompt_text.strip():
        raise SystemExit("--prompt-text is required for zero_shot mode. Use the transcript of --prompt-wav.")

    cosyvoice = AutoModel(
        model_dir=str(model_dir),
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )

    tts_text = args.text
    prompt_text = args.prompt_text
    instruct_text = args.instruct_text
    if uses_cosyvoice3(model_dir):
        if args.mode == "zero_shot":
            prompt_text = ensure_cosyvoice3_prefix(prompt_text)
        elif args.mode == "cross_lingual":
            tts_text = ensure_cosyvoice3_prefix(tts_text)
        else:
            instruct_text = ensure_cosyvoice3_prefix(instruct_text)

    if args.mode == "zero_shot":
        results = cosyvoice.inference_zero_shot(
            tts_text,
            prompt_text,
            str(prompt_wav),
            stream=args.stream,
            speed=args.speed,
            text_frontend=args.text_frontend,
        )
    elif args.mode == "cross_lingual":
        results = cosyvoice.inference_cross_lingual(
            tts_text,
            str(prompt_wav),
            stream=args.stream,
            speed=args.speed,
            text_frontend=args.text_frontend,
        )
    else:
        results = cosyvoice.inference_instruct2(
            tts_text,
            instruct_text,
            str(prompt_wav),
            stream=args.stream,
            speed=args.speed,
            text_frontend=args.text_frontend,
        )

    for idx, item in enumerate(results):
        output_path = output_dir / "{}_{}_{}.wav".format(args.output_prefix, args.mode, idx)
        torchaudio.save(str(output_path), item["tts_speech"], cosyvoice.sample_rate)
        print("wrote {}".format(output_path))


if __name__ == "__main__":
    main()

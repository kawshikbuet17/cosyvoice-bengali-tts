#!/usr/bin/env python3
"""Bengali emotion/style inference helper for CosyVoice3."""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402
from emotion_tags import (  # noqa: E402
    DEFAULT_BASE_INSTRUCTION,
    DEFAULT_NEUTRAL_INSTRUCTION,
    convert_tagged_text,
    ensure_cosyvoice3_prefix,
)
import torchaudio  # noqa: E402


DEFAULT_TEXT = "[warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।"
DEFAULT_INSTRUCT = ensure_cosyvoice3_prefix(DEFAULT_NEUTRAL_INSTRUCTION)


def uses_cosyvoice3(model_dir):
    return "cosyvoice3" in str(model_dir).lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bengali emotion/style inference with a CosyVoice3 model directory.")
    parser.add_argument("--model-dir", default="../../../pretrained_models/Fun-CosyVoice3-0.5B",
                        help="CosyVoice3 model directory containing cosyvoice3.yaml, llm.pt, flow.pt, hift.pt, and ONNX files")
    parser.add_argument("--mode", choices=["cross_lingual", "instruct2"], default="cross_lingual",
                        help="Inference mode. cross_lingual is the recommended first smoke test path.")
    parser.add_argument("--text", default=DEFAULT_TEXT,
                        help="Bengali text to synthesize. Leading emotion tags such as [warm][smile] are supported.")
    parser.add_argument("--prompt-wav", required=True, help="Prompt audio path used for speaker/style reference")
    parser.add_argument("--instruct-text", default="",
                        help="Optional explicit instruction text for instruct2 mode. If omitted and --text starts with tags, instruction is derived from the tags.")
    parser.add_argument("--base-instruct", default=DEFAULT_BASE_INSTRUCTION,
                        help="Base instruction used when converting bracket tags into CosyVoice3 instruct text")
    parser.add_argument("--output-dir", default="outputs/bengali_emotion_inference", help="Directory for generated wav files")
    parser.add_argument("--output-prefix", default="bengali_emotion", help="Generated wav filename prefix")
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


def prepare_inputs(raw_text, instruct_text, base_instruct, mode):
    converted = convert_tagged_text(raw_text, base_instruction=base_instruct)
    clean_text = converted["text"] or raw_text.strip()

    if mode == "cross_lingual":
        if converted["tags"]:
            tts_text = converted["instruction"] + clean_text
            print("Info: converted leading tags into cross-lingual instruction prefix.")
        else:
            tts_text = ensure_cosyvoice3_prefix(clean_text)
        return tts_text, "", converted

    if instruct_text.strip():
        return clean_text, ensure_cosyvoice3_prefix(instruct_text.strip()), converted

    if converted["tags"]:
        print("Info: converted leading tags into instruct2 instruction text.")
        return clean_text, converted["instruction"], converted

    return clean_text, ensure_cosyvoice3_prefix(base_instruct), converted


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
    if not uses_cosyvoice3(model_dir):
        raise SystemExit("This helper expects a CosyVoice3 model directory.")

    cosyvoice = AutoModel(
        model_dir=str(model_dir),
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
    )

    tts_text, instruct_text, converted = prepare_inputs(
        args.text,
        args.instruct_text,
        args.base_instruct,
        args.mode,
    )
    if converted["unsupported_tags"]:
        print("Warning: unsupported tags ignored: {}".format(", ".join(converted["unsupported_tags"])))

    if args.mode == "cross_lingual":
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
            instruct_text or DEFAULT_INSTRUCT,
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

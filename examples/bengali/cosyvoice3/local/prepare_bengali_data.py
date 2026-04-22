#!/usr/bin/env python3
"""Prepare raw Bengali .flac + .json data for CosyVoice3.

This script converts the project dataset contract into the Kaldi-style files
used by CosyVoice tools:

  wav.scp, text, utt2spk, spk2utt, and optional instruct

It does not require the real dataset to be present locally. Run it on the
remote Linux server where the dataset exists.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys


DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Bengali raw .flac + .json data for CosyVoice3."
    )
    parser.add_argument("--dataset-root", required=True, help="Raw dataset root, e.g. /home/kawshik/TTS_Dataset")
    parser.add_argument("--output-dir", default="data", help="Output data directory, usually examples/bengali/cosyvoice3/data")
    parser.add_argument("--prefix", default="bengali", help="Split directory prefix, e.g. bengali -> bengali_train")
    parser.add_argument("--speaker-id", default=None, help="Keep exactly one speaker_id for a single-speaker smoke test")
    parser.add_argument("--speaker-ids", nargs="*", default=None, help="Optional explicit list of speaker IDs to keep")
    parser.add_argument("--min-utterances-per-speaker", type=int, default=1, help="Drop speakers with fewer kept utterances")
    parser.add_argument("--max-speakers", type=int, default=None, help="Keep only the top N speakers by utterance count")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation ratio per speaker")
    parser.add_argument("--test-ratio", type=float, default=0.02, help="Test ratio per speaker")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for splits")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum duration in seconds when JSON duration is present")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum duration in seconds when JSON duration is present")
    parser.add_argument("--max-text-chars", type=int, default=250, help="Maximum transcript length in characters")
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT, help="CosyVoice3 instruct text written to instruct files")
    parser.add_argument("--no-instruct", action="store_true", help="Do not write instruct files")
    parser.add_argument("--convert-to-wav", action="store_true", help="Convert selected audio to 24 kHz mono WAV with ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target WAV sample rate when --convert-to-wav is used")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite converted WAV files")
    return parser.parse_args()


def normalize_text(text):
    text = text.replace("\u200c", "")
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def sanitize_id(value):
    value = str(value).strip()
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def load_json(path, skipped):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        skipped.append((str(path), "malformed_json", str(exc)))
        return None


def extract_sentences(data):
    annotation = data.get("annotation")
    if isinstance(annotation, dict):
        annotation = [annotation]
    if not isinstance(annotation, list):
        return ""

    sentences = []
    for item in annotation:
        if not isinstance(item, dict):
            continue
        sentence = item.get("sentence")
        if isinstance(sentence, str):
            sentence = normalize_text(sentence)
            if sentence:
                sentences.append(sentence)
    return normalize_text(" ".join(sentences))


def infer_speaker_from_path(json_path, dataset_root):
    try:
        parts = json_path.relative_to(dataset_root).parts
    except ValueError:
        parts = json_path.parts
    for marker in ("Male", "Female"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def build_flac_indexes(flac_paths):
    by_same_stem = defaultdict(list)
    by_name = defaultdict(list)
    for path in flac_paths:
        by_same_stem[path.stem].append(path)
        by_name[path.name].append(path)
    return by_same_stem, by_name


def resolve_audio_path(data, json_path, dataset_root, flac_by_stem, flac_by_name):
    same_dir_audio = json_path.with_suffix(".flac")
    if same_dir_audio.exists():
        return same_dir_audio

    json_audio_path = data.get("path")
    if isinstance(json_audio_path, str) and json_audio_path.strip():
        raw = json_audio_path.strip()
        candidates = [
            dataset_root / raw,
            json_path.parent / raw,
            dataset_root / raw.lstrip("/"),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.suffix.lower() == ".flac":
                return candidate

        basename_matches = flac_by_name.get(Path(raw).name, [])
        if len(basename_matches) == 1:
            return basename_matches[0]

    speech_id = data.get("speech_id")
    for key in (json_path.stem, speech_id):
        if isinstance(key, str) and key:
            matches = flac_by_stem.get(key, [])
            if len(matches) == 1:
                return matches[0]
    return None


def valid_duration(duration, args):
    if duration is None:
        return True
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return True
    return args.min_duration <= duration <= args.max_duration


def select_speakers(records, args, skipped):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["speaker_id"]].append(record)

    selected = set(grouped.keys())
    explicit = set()
    if args.speaker_id:
        explicit.add(args.speaker_id)
    if args.speaker_ids:
        explicit.update(args.speaker_ids)
    if explicit:
        selected = selected.intersection(explicit)

    if args.min_utterances_per_speaker > 1:
        selected = {
            spk for spk in selected
            if len(grouped[spk]) >= args.min_utterances_per_speaker
        }

    ranked = sorted(selected, key=lambda spk: (-len(grouped[spk]), spk))
    if args.max_speakers is not None:
        ranked = ranked[:args.max_speakers]
    selected = set(ranked)

    final_records = []
    for spk, spk_records in grouped.items():
        if spk in selected:
            final_records.extend(spk_records)
        else:
            reason = "speaker_filtered_or_too_few_utterances"
            for record in spk_records:
                skipped.append((record["json_path"], reason, spk))
    return final_records, ranked


def split_count(total, ratio, reserve):
    if ratio <= 0 or total <= reserve:
        return 0
    count = int(round(total * ratio))
    if count == 0:
        return 0
    return min(count, max(0, total - reserve))


def split_records(records, args):
    rng = random.Random(args.seed)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["speaker_id"]].append(record)

    splits = {"train": [], "dev": [], "test": []}
    for speaker_id in sorted(grouped):
        spk_records = list(grouped[speaker_id])
        rng.shuffle(spk_records)
        total = len(spk_records)
        n_test = split_count(total, args.test_ratio, reserve=2)
        n_val = split_count(total - n_test, args.val_ratio, reserve=1)
        test = spk_records[:n_test]
        dev = spk_records[n_test:n_test + n_val]
        train = spk_records[n_test + n_val:]
        if not train and (dev or test):
            train.append((dev or test).pop())
        splits["train"].extend(train)
        splits["dev"].extend(dev)
        splits["test"].extend(test)
    return splits


def make_utt_ids(records):
    seen = Counter()
    for record in records:
        speech_id = record.get("speech_id") or Path(record["json_path"]).stem
        base = sanitize_id("{}_{}".format(record["speaker_id"], speech_id))
        seen[base] += 1
        record["utt"] = base if seen[base] == 1 else "{}_{}".format(base, seen[base])


def run_ffmpeg(src, dst, sample_rate, overwrite):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False, "ffmpeg_not_found"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return True, "exists"
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sample_rate),
        str(dst),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    if process.returncode != 0:
        return False, stderr.decode("utf-8", errors="replace")[-500:]
    return True, "converted"


def maybe_convert_audio(records, output_dir, args, skipped):
    if not args.convert_to_wav:
        for record in records:
            record["prepared_audio_path"] = record["audio_path"]
        return records

    converted = []
    audio_root = output_dir / "audio_wav_{}hz".format(args.sample_rate)
    for record in records:
        rel = Path(sanitize_id(record["speaker_id"])) / "{}.wav".format(record["utt"])
        wav_path = audio_root / rel
        ok, status = run_ffmpeg(record["audio_path"], wav_path, args.sample_rate, args.overwrite)
        if ok:
            record["prepared_audio_path"] = wav_path
            converted.append(record)
        else:
            skipped.append((record["json_path"], "ffmpeg_failed", status))
    return converted


def write_split(split_name, records, output_dir, prefix, instruct, no_instruct):
    split_dir = output_dir / "{}_{}".format(prefix, split_name)
    split_dir.mkdir(parents=True, exist_ok=True)

    spk2utt = defaultdict(list)
    with (split_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
            (split_dir / "text").open("w", encoding="utf-8") as text_f, \
            (split_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk_f:
        instruct_f = None
        if not no_instruct:
            instruct_f = (split_dir / "instruct").open("w", encoding="utf-8")
        try:
            for record in records:
                utt = record["utt"]
                spk = record["speaker_id"]
                audio_path = Path(record["prepared_audio_path"]).resolve()
                wav_f.write("{} {}\n".format(utt, audio_path))
                text_f.write("{} {}\n".format(utt, record["text"]))
                utt2spk_f.write("{} {}\n".format(utt, spk))
                if instruct_f is not None:
                    instruct_f.write("{} {}\n".format(utt, instruct))
                spk2utt[spk].append(utt)
        finally:
            if instruct_f is not None:
                instruct_f.close()

    with (split_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write("{} {}\n".format(spk, " ".join(spk2utt[spk])))
    return split_dir


def write_reports(output_dir, args, report, speaker_counts, skipped):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bengali_prep_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (output_dir / "speaker_stats.tsv").open("w", encoding="utf-8") as f:
        f.write("speaker_id\tutterances\n")
        for speaker_id, count in sorted(speaker_counts.items(), key=lambda item: (-item[1], item[0])):
            f.write("{}\t{}\n".format(speaker_id, count))
    with (output_dir / "skipped_records.tsv").open("w", encoding="utf-8") as f:
        f.write("path\treason\tdetail\n")
        for path, reason, detail in skipped:
            detail = str(detail).replace("\n", " ").replace("\t", " ")
            f.write("{}\t{}\t{}\n".format(path, reason, detail))


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit("Dataset root does not exist: {}".format(dataset_root))
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("Split ratios must be non-negative and sum to less than 1.")

    skipped = []
    json_paths = sorted(dataset_root.rglob("*.json"))
    flac_paths = sorted(dataset_root.rglob("*.flac"))
    flac_by_stem, flac_by_name = build_flac_indexes(flac_paths)

    records = []
    for json_path in json_paths:
        data = load_json(json_path, skipped)
        if data is None:
            continue

        speaker_id = data.get("speaker_id") or infer_speaker_from_path(json_path, dataset_root)
        if not speaker_id:
            skipped.append((str(json_path), "missing_speaker_id", ""))
            continue
        speaker_id = str(speaker_id).strip()

        text = extract_sentences(data)
        if not text:
            skipped.append((str(json_path), "empty_sentence", ""))
            continue
        if args.max_text_chars and len(text) > args.max_text_chars:
            skipped.append((str(json_path), "text_too_long", len(text)))
            continue
        if not valid_duration(data.get("duration"), args):
            skipped.append((str(json_path), "duration_out_of_range", data.get("duration")))
            continue

        audio_path = resolve_audio_path(data, json_path, dataset_root, flac_by_stem, flac_by_name)
        if audio_path is None:
            skipped.append((str(json_path), "missing_matching_flac", ""))
            continue
        if " " in str(audio_path):
            skipped.append((str(json_path), "audio_path_contains_space", str(audio_path)))
            continue

        records.append({
            "json_path": str(json_path),
            "audio_path": audio_path,
            "speaker_id": speaker_id,
            "speech_id": data.get("speech_id") or json_path.stem,
            "duration": data.get("duration"),
            "gender": data.get("gender"),
            "text": text,
        })

    records, selected_speakers = select_speakers(records, args, skipped)
    make_utt_ids(records)
    records = maybe_convert_audio(records, output_dir, args, skipped)
    splits = split_records(records, args)

    split_dirs = {}
    for split_name in ("train", "dev", "test"):
        split_dirs[split_name] = write_split(
            split_name,
            splits[split_name],
            output_dir,
            args.prefix,
            args.instruct,
            args.no_instruct,
        )

    speaker_counts = Counter(record["speaker_id"] for record in records)
    skipped_counter = Counter(item[1] for item in skipped)
    report = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "json_files_found": len(json_paths),
        "flac_files_found": len(flac_paths),
        "records_kept": len(records),
        "speakers_kept": selected_speakers,
        "speaker_count": len(selected_speakers),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "split_dirs": {name: str(path) for name, path in split_dirs.items()},
        "skipped_total": len(skipped),
        "skipped_by_reason": dict(sorted(skipped_counter.items())),
        "convert_to_wav": args.convert_to_wav,
        "sample_rate": args.sample_rate,
    }
    write_reports(output_dir, args, report, speaker_counts, skipped)

    print("JSON files found: {}".format(len(json_paths)))
    print("FLAC files found: {}".format(len(flac_paths)))
    print("Speakers kept: {}".format(len(selected_speakers)))
    if len(selected_speakers) == 1:
        print("WARNING: only one speaker was kept; this is useful for debugging but not true multi-speaker training.")
    print("Records kept: {}".format(len(records)))
    print("Train/dev/test: {}/{}/{}".format(len(splits["train"]), len(splits["dev"]), len(splits["test"])))
    print("Skipped: {}".format(len(skipped)))
    print("Wrote metadata:")
    for split_name in ("train", "dev", "test"):
        print("  {}: {}".format(split_name, split_dirs[split_name]))
    print("Report: {}".format(output_dir / "bengali_prep_report.json"))

    if not records:
        return 1
    if not splits["train"]:
        print("ERROR: train split is empty.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

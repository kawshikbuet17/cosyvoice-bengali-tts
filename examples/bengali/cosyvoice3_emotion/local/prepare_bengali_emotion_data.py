#!/usr/bin/env python3
"""Prepare an emotion-tagged Bengali dataset for CosyVoice3 training."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

from emotion_tags import (
    DEFAULT_BASE_INSTRUCTION,
    DEFAULT_NEUTRAL_INSTRUCTION,
    convert_tagged_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an emotion-tagged Bengali .txt + .wav dataset for CosyVoice3."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root, e.g. /home/kawshik/TTS_Dataset_Ghorer_Bazar")
    parser.add_argument("--output-dir", default="data", help="Output data directory")
    parser.add_argument("--prefix", default="bengali_emotion", help="Split directory prefix")
    parser.add_argument("--speaker-id", default=None, help="Keep exactly one speaker id for smoke testing")
    parser.add_argument("--speaker-ids", nargs="*", default=None, help="Optional explicit list of speaker ids to keep")
    parser.add_argument("--male-speaker-id", default="emotion_male", help="Speaker id assigned to *_male.wav")
    parser.add_argument("--female-speaker-id", default="emotion_female", help="Speaker id assigned to *_female.wav")
    parser.add_argument("--min-utterances-per-speaker", type=int, default=1, help="Drop speakers with fewer kept utterances")
    parser.add_argument("--max-speakers", type=int, default=None, help="Keep only the top N speakers by utterance count")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio per speaker")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test ratio per speaker")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for splits")
    parser.add_argument("--max-text-chars", type=int, default=250, help="Maximum transcript length in characters")
    parser.add_argument("--base-instruct", default=DEFAULT_BASE_INSTRUCTION, help="Base CosyVoice3 instruction prefix")
    parser.add_argument("--neutral-instruct", default=DEFAULT_NEUTRAL_INSTRUCTION, help="Fallback instruction fragment for tagless lines")
    parser.add_argument("--no-instruct", action="store_true", help="Do not write instruct files")
    parser.add_argument("--convert-to-wav", action="store_true", help="Convert selected audio to 24 kHz mono wav with ffmpeg")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target wav sample rate when --convert-to-wav is used")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite converted wav files")
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


def load_text(path, skipped):
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        skipped.append((str(path), "malformed_text", str(exc)))
        return None


def valid_text_length(text, args):
    if not text:
        return False
    if args.max_text_chars and len(text) > args.max_text_chars:
        return False
    return True


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
            for record in spk_records:
                skipped.append((record["text_path"], "speaker_filtered_or_too_few_utterances", spk))
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
        base = sanitize_id("{}_{}_{}".format(record["speaker_id"], record["source_id"], record["gender"]))
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
            skipped.append((record["text_path"], "ffmpeg_failed", status))
    return converted


def write_split(split_name, records, output_dir, prefix, no_instruct):
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
                    instruct_f.write("{} {}\n".format(utt, record["instruct"]))
                spk2utt[spk].append(utt)
        finally:
            if instruct_f is not None:
                instruct_f.close()

    with (split_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt):
            f.write("{} {}\n".format(spk, " ".join(spk2utt[spk])))
    return split_dir


def write_reports(output_dir, report, speaker_counts, tag_counts, unsupported_counts, skipped):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bengali_emotion_prep_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (output_dir / "speaker_stats.tsv").open("w", encoding="utf-8") as f:
        f.write("speaker_id\tutterances\n")
        for speaker_id, count in sorted(speaker_counts.items(), key=lambda item: (-item[1], item[0])):
            f.write("{}\t{}\n".format(speaker_id, count))
    with (output_dir / "tag_stats.tsv").open("w", encoding="utf-8") as f:
        f.write("tag\tcount\n")
        for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
            f.write("{}\t{}\n".format(tag, count))
    with (output_dir / "unsupported_tag_stats.tsv").open("w", encoding="utf-8") as f:
        f.write("tag\tcount\n")
        for tag, count in sorted(unsupported_counts.items(), key=lambda item: (-item[1], item[0])):
            f.write("{}\t{}\n".format(tag, count))
    with (output_dir / "skipped_records.tsv").open("w", encoding="utf-8") as f:
        f.write("path\treason\tdetail\n")
        for path, reason, detail in skipped:
            detail = str(detail).replace("\n", " ").replace("\t", " ")
            f.write("{}\t{}\t{}\n".format(path, reason, detail))


def iter_candidate_bases(dataset_root):
    for txt_path in sorted(dataset_root.rglob("*.txt")):
        stem = txt_path.stem
        male_wav = txt_path.with_name(stem + "_male.wav")
        female_wav = txt_path.with_name(stem + "_female.wav")
        yield txt_path, {"male": male_wav, "female": female_wav}


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_root.exists():
        raise SystemExit("Dataset root does not exist: {}".format(dataset_root))
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("Split ratios must be non-negative and sum to less than 1.")

    skipped = []
    tag_counts = Counter()
    unsupported_counts = Counter()
    txt_paths = sorted(dataset_root.rglob("*.txt"))
    wav_paths = sorted(dataset_root.rglob("*.wav"))

    records = []
    for txt_path, gender_map in iter_candidate_bases(dataset_root):
        raw_text = load_text(txt_path, skipped)
        if raw_text is None:
            continue

        converted = convert_tagged_text(
            raw_text,
            base_instruction=args.base_instruct,
            neutral_instruction=args.neutral_instruct,
        )
        text = normalize_text(converted["text"])
        if not valid_text_length(text, args):
            skipped.append((str(txt_path), "text_empty_or_too_long", len(text)))
            continue

        for tag in converted["tags"]:
            tag_counts[tag] += 1
        for tag in converted["unsupported_tags"]:
            unsupported_counts[tag] += 1

        found_any_audio = False
        for gender, wav_path in gender_map.items():
            if not wav_path.exists():
                continue
            found_any_audio = True
            speaker_id = args.male_speaker_id if gender == "male" else args.female_speaker_id
            if " " in str(wav_path):
                skipped.append((str(txt_path), "audio_path_contains_space", str(wav_path)))
                continue
            records.append({
                "text_path": str(txt_path),
                "audio_path": wav_path,
                "speaker_id": speaker_id,
                "source_id": txt_path.stem,
                "gender": gender,
                "raw_text": raw_text,
                "text": text,
                "instruct": converted["instruction"],
                "tags": converted["tags"],
                "unsupported_tags": converted["unsupported_tags"],
            })
        if not found_any_audio:
            skipped.append((str(txt_path), "missing_matching_wav", "expected *_male.wav or *_female.wav"))

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
            args.no_instruct,
        )

    speaker_counts = Counter(record["speaker_id"] for record in records)
    skipped_counter = Counter(item[1] for item in skipped)
    report = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "text_files_found": len(txt_paths),
        "wav_files_found": len(wav_paths),
        "records_kept": len(records),
        "speakers_kept": selected_speakers,
        "speaker_count": len(selected_speakers),
        "split_counts": {name: len(items) for name, items in splits.items()},
        "split_dirs": {name: str(path) for name, path in split_dirs.items()},
        "tags_seen": dict(sorted(tag_counts.items())),
        "unsupported_tags": dict(sorted(unsupported_counts.items())),
        "skipped_total": len(skipped),
        "skipped_by_reason": dict(sorted(skipped_counter.items())),
        "convert_to_wav": args.convert_to_wav,
        "sample_rate": args.sample_rate,
    }
    write_reports(output_dir, report, speaker_counts, tag_counts, unsupported_counts, skipped)

    print("Text files found: {}".format(len(txt_paths)))
    print("WAV files found: {}".format(len(wav_paths)))
    print("Speakers kept: {}".format(len(selected_speakers)))
    print("Records kept: {}".format(len(records)))
    print("Train/dev/test: {}/{}/{}".format(len(splits["train"]), len(splits["dev"]), len(splits["test"])))
    print("Skipped: {}".format(len(skipped)))
    print("Wrote metadata:")
    for split_name in ("train", "dev", "test"):
        print("  {}: {}".format(split_name, split_dirs[split_name]))
    print("Report: {}".format(output_dir / "bengali_emotion_prep_report.json"))

    if not records:
        return 1
    if not splits["train"]:
        print("ERROR: train split is empty.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

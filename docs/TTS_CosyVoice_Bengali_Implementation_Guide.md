# TTS CosyVoice Bengali Implementation Guide

This project adapts the FunAudioLLM/CosyVoice repository for Bangla/Bengali TTS work. The target input is a raw Bengali speech dataset with matching `.flac` audio and `.json` metadata, and the target output is a CosyVoice3-based Bengali TTS workflow that can prepare data, fine-tune a model, and synthesize Bengali speech.

This guide explains the Bengali implementation scaffold in this fork and gives a beginner-friendly sequential runbook. The current project direction is **CosyVoice3 first**. CosyVoice2 may still be useful as a reference, but Bengali implementation should be built against CosyVoice3 unless a real blocker appears.

For a beginner explanation of TTS and CosyVoice concepts, see [TTS_CosyVoice_Beginner_Guide.md](TTS_CosyVoice_Beginner_Guide.md).

For a concise high-level summary of how this fork differs from original CosyVoice, see [Major_Changes_From_Original_CosyVoice.md](Major_Changes_From_Original_CosyVoice.md).

Important current status:

```text
This document is an implementation guide and runbook.
The Bengali CosyVoice3 scaffold has been added under examples/bengali/cosyvoice3.
Full execution requires the real remote dataset and downloaded pretrained model files. After training starts, the next validation target is checkpoint-based Bengali inference.
```

The real Bengali dataset is on the remote Linux server. The local repository should not assume direct access to that dataset.

## 1. Purpose

The purpose of this fork is to add a Bengali data bridge and Bengali running guide around the original CosyVoice codebase.

The main goal is:

```text
raw Bengali .flac + .json dataset
  -> CosyVoice metadata files
  -> speaker embeddings and speech tokens
  -> parquet training data
  -> CosyVoice3 fine-tuning
  -> Bengali speech inference
```

Main design choices:

- Keep the original CosyVoice model code as intact as possible.
- Use CosyVoice3 as the primary target.
- Reuse existing CosyVoice tools for embeddings, speech tokens, parquet creation, and training.
- Add Bengali-specific dataset preparation instead of rewriting the full training system.
- Support single-speaker debugging first, then multi-speaker scaling.
- Keep `.flac` support if the audio backend can read it; add optional conversion only if needed.
- Document every step clearly for beginner sequential execution.

## 2. Why CosyVoice3 Is The First Target

CosyVoice3 is the newest model family in this repository and is designed for stronger multilingual, cross-lingual, and instruction-guided speech generation.

The Bengali adaptation should use CosyVoice3 first because:

- the original repo recommends Fun-CosyVoice3 for better performance;
- CosyVoice3 has a dedicated training example under `examples/libritts/cosyvoice3`;
- the training path already shows how to prepare metadata, extract features, create parquet data, and launch training;
- Bengali work should build around that existing path instead of creating a separate framework.

CosyVoice2 can remain a fallback or comparison target, but the first Bengali implementation should be:

```text
examples/bengali/cosyvoice3/
```

## 3. Single-Speaker And Multi-Speaker Strategy

CosyVoice naturally uses speaker metadata through:

```text
utt2spk
spk2utt
```

Because of that, one Bengali preparation script can support both single-speaker and multi-speaker preparation.

However, the documentation and run commands should clearly separate the two workflows.

### Single-Speaker Purpose

Single-speaker preparation is the safest first debugging path.

Its purpose is to prove the full Bengali pipeline end to end:

```text
one selected speaker
  -> metadata generation
  -> embedding/token extraction
  -> parquet generation
  -> training starts
  -> inference can be tested
```

This is useful because if something fails, the problem space is smaller.

Example argument:

```bash
--speaker-id 01332512906
```

This means:

```text
scan the full raw dataset
keep only JSON records where speaker_id == 01332512906
skip all other speakers
write CosyVoice metadata for that one speaker
```

### Multi-Speaker Purpose

Multi-speaker preparation is the real scalable direction after the single-speaker pipeline is stable.

It should allow:

- multiple selected speakers;
- minimum utterance count per speaker;
- maximum number of speakers for early experiments;
- train/dev/test splits without speaker metadata loss;
- speaker distribution reports.

Example arguments:

```bash
--min-utterances-per-speaker 200
--max-speakers 10
```

This means:

```text
keep speakers with at least 200 valid utterances
choose up to 10 speakers
prepare one multi-speaker CosyVoice dataset
```

## 4. Original CosyVoice Dataset Pattern Vs Bengali Dataset Pattern

The original CosyVoice examples do not train directly from raw annotation JSON files. They first convert a dataset into Kaldi-style metadata and then into parquet training data.

### Original CosyVoice Example Pattern

The LibriTTS example prepares data like this:

```text
LibriTTS audio/text
  -> wav.scp
  -> text
  -> utt2spk
  -> spk2utt
  -> utt2embedding.pt / spk2embedding.pt
  -> utt2speech_token.pt
  -> parquet files
  -> data.list
  -> training
```

Typical CosyVoice metadata files:

```text
wav.scp
text
utt2spk
spk2utt
```

Example:

```text
wav.scp:
utt_000001 /path/to/audio.wav

text:
utt_000001 This is the transcript.

utt2spk:
utt_000001 speaker_001

spk2utt:
speaker_001 utt_000001
```

### Bengali Raw Dataset Pattern

The Bengali dataset starts differently:

```text
/home/kawshik/TTS_Dataset/
  Male/
    <speaker_id>/
      optional_nested_folder/
        utterance.flac
        utterance.json
  Female/
    <speaker_id>/
      optional_nested_folder/
        utterance.flac
        utterance.json
```

The transcript is inside JSON:

```text
annotation[*]["sentence"]
```

The Bengali preparation layer must bridge this difference:

```text
raw Bengali .flac + .json
  -> CosyVoice wav.scp/text/utt2spk/spk2utt
```

After that bridge is built, the original CosyVoice tools can take over.

## 5. Dataset Contract

The raw dataset root on the remote server is expected to look approximately like this:

```text
/home/kawshik/TTS_Dataset/
  Male/
    01332512906/
      2024-05-07/
        00d4da35-bebc-471b-aa92-0d9398388e98.flac
        00d4da35-bebc-471b-aa92-0d9398388e98.json
  Female/
    ...
```

Extra nested folders are allowed. The preparation script should scan recursively and should not assume a fixed folder depth.

Each valid utterance should have:

- one `.flac` audio file;
- one matching `.json` metadata file;
- a `speaker_id`;
- transcript text under `annotation[*]["sentence"]`.

Example JSON shape with sentence text and word-level timing:

```json
{
    "duration": 6.66,
    "speaker_id": "01332512906",
    "gender": "পুরুষ",
    "script_source": "manually curated",
    "path": "gold/create/sentence+word/01332512906/2024-05-07/00d4da35-bebc-471b-aa92-0d9398388e98.flac",
    "speech_id": "00d4da35-bebc-471b-aa92-0d9398388e98",
    "annotation": [
        {
            "tagList": [],
            "start": 1.07355922,
            "end": 6.08486152,
            "id": "rPboUgsw",
            "sentence": "তার কথাগুলো শুনে বুঝলাম  বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।",
            "words": [
                {
                    "start": 1.17355922,
                    "end": 1.43355922,
                    "id": "rPboUgsw-1",
                    "word": "তার"
                },
                {
                    "start": 1.43355922,
                    "end": 1.99355922,
                    "id": "rPboUgsw-2",
                    "word": "কথাগুলো"
                },
                {
                    "start": 1.99355922,
                    "end": 2.25355922,
                    "id": "rPboUgsw-3",
                    "word": "শুনে"
                },
                {
                    "start": 2.25355922,
                    "end": 2.79355922,
                    "id": "rPboUgsw-4",
                    "word": "বুঝলাম"
                },
                {
                    "start": 2.79355922,
                    "end": 3.31355922,
                    "id": "rPboUgsw-5",
                    "word": "বয়সের"
                },
                {
                    "start": 3.31355922,
                    "end": 3.75355922,
                    "id": "rPboUgsw-6",
                    "word": "তুলনায়"
                },
                {
                    "start": 3.75355922,
                    "end": 3.95355922,
                    "id": "rPboUgsw-7",
                    "word": "সে"
                },
                {
                    "start": 3.95355922,
                    "end": 4.71355922,
                    "id": "rPboUgsw-8",
                    "word": "মানসিকতায়"
                },
                {
                    "start": 4.71355922,
                    "end": 5.03355922,
                    "id": "rPboUgsw-9",
                    "word": "অনেক"
                },
                {
                    "start": 5.03355922,
                    "end": 5.31355922,
                    "id": "rPboUgsw-10",
                    "word": "বড়"
                },
                {
                    "start": 5.31355922,
                    "end": 5.53355922,
                    "id": "rPboUgsw-11",
                    "word": "হয়ে"
                },
                {
                    "start": 5.53355922,
                    "end": 6.04355922,
                    "id": "rPboUgsw-12",
                    "word": "গিয়েছে"
                }
            ]
        }
    ]
}
```

### Which raw Bengali dataset fields are actually used now

The current Bengali preparation script uses the raw dataset conservatively.

Required or effectively required:

- one real `.flac` audio file;
- one matching `.json` file;
- transcript text from `annotation[*]["sentence"]`;
- `speaker_id`, or a folder path from which the speaker ID can be inferred.

Used when present:

- `path`: helps resolve the correct `.flac` file if simple same-folder or same-stem matching is not enough;
- `duration`: used for duration filtering against `--min-duration` and `--max-duration`;
- `speech_id`: used when building stable utterance IDs;
- `gender`: retained in the prepared record/report, but not used to control training.

Currently not used for first-pass training decisions:

- `script_source`;
- `annotation[*].tagList`;
- `annotation[*].start` and `annotation[*].end`;
- `annotation[*].id`;
- `annotation[*].words[*]` word-level timing metadata.

So the first Bengali CosyVoice3 baseline mainly depends on clean audio-text pairing and correct speaker identity. Richer JSON fields are preserved for future alignment or dataset-quality work, but they are not the main drivers of the current training pipeline.

## 6. Sample Dummy Dataset

A documentation-only sample raw dataset is included at:

```text
docs/sample_bengali_raw_dataset/
```

It shows the expected folder pattern with:

```text
Male speakers:   2
Female speakers: 2
```

The sample uses `.flac.placeholder` files instead of real audio. It is meant to explain the structure only; do not use it for training or feature extraction.

## 7. Files Added

These files were added for the Bengali CosyVoice3 scaffold.

```text
examples/bengali/cosyvoice3/
  run.sh
  path.sh
  conf/
    cosyvoice3_bengali.yaml
    ds_stage2.json
  local/
    prepare_bengali_data.py
    infer_bengali.py

.gitattributes

docs/
  TTS_CosyVoice_Beginner_Guide.md
  TTS_CosyVoice_Bengali_Implementation_Guide.md
  Major_Changes_From_Original_CosyVoice.md
  sample_bengali_raw_dataset/
    README.md
```

Recommended tracked addition:

```text
environment_snapshots/
  README.md
  conda_list_server_working.txt
  requirements_server_working.txt
  python_version.txt
  nvidia_smi_server_working.txt
  torch_cuda_check.txt
  system_info.txt
```

## 8. Files Modified

The implementation still follows the original CosyVoice structure, but several practical changes were added after real remote-server execution.

Current modified files and purpose:

- `README.md`
  Add a short note that this fork is adapted for Bengali CosyVoice3 TTS and point users to the Bengali documentation.

- `.gitattributes`
  Force Linux-friendly LF line endings for shell/Python/config/docs files. This prevents server errors such as `python3\r` after moving files from Windows to Linux.

- `examples/bengali/cosyvoice3/run.sh`
  Add the Bengali staged runner, dataset arguments, `--speaker_ids`, GPU selection through `--cuda_visible_devices`, and `--checkpoint` support for resuming training from a selected checkpoint.

- `examples/bengali/cosyvoice3/conf/cosyvoice3_bengali.yaml`
  Add the Bengali CosyVoice3 training config. The config now includes `save_interval_epochs: 10` so future training does not save a huge checkpoint every epoch.

- `examples/bengali/cosyvoice3/local/prepare_bengali_data.py`
  Convert the raw Bengali `.flac` + `.json` dataset into CosyVoice metadata files.

- `examples/bengali/cosyvoice3/local/infer_bengali.py`
  Add terminal inference for Bengali testing. It supports `zero_shot`, `cross_lingual`, and `instruct2`, and now automatically inserts the CosyVoice3 `<|endofprompt|>` system separator when needed.

- `cosyvoice/utils/executor.py`
  Add checkpoint save interval support through `save_interval_epochs` to reduce disk usage during long training.

- `webui.py`
  Add an English/Bangla Gradio UI, Bengali usage instructions, a prepared-by footer, and CosyVoice3 prompt-prefix handling for inference.

- `docs/`
  Add Bengali-specific beginner, implementation, major-change, and sample-dataset documentation.

Core model architecture is not broadly refactored. The changes are limited to a Bengali adaptation path, running scripts, inference helpers, checkpoint practicality, and documentation.

## 9. Bengali Dataset Preparation Design

The Bengali preparation script creates CosyVoice metadata files.

Script path:

```text
examples/bengali/cosyvoice3/local/prepare_bengali_data.py
```

The script supports:

- `--dataset-root`
- `--output-dir`
- `--speaker-id`
- `--speaker-ids`
- `--min-utterances-per-speaker`
- `--max-speakers`
- `--val-ratio`
- `--test-ratio`
- `--seed`
- `--min-duration`
- `--max-duration`
- `--max-text-chars`
- `--instruct`
- optional `--convert-to-wav`
- optional `--sample-rate 24000`

The script should write:

```text
data/bengali_train/wav.scp
data/bengali_train/text
data/bengali_train/utt2spk
data/bengali_train/spk2utt

data/bengali_dev/wav.scp
data/bengali_dev/text
data/bengali_dev/utt2spk
data/bengali_dev/spk2utt

data/bengali_test/wav.scp
data/bengali_test/text
data/bengali_test/utt2spk
data/bengali_test/spk2utt
```

It also writes reports:

```text
bengali_prep_report.json
speaker_stats.tsv
skipped_records.tsv
```

## 10. Transcript Extraction

Transcript extraction should be defensive.

Primary source:

```text
annotation[*]["sentence"]
```

Recommended behavior:

1. Load JSON safely.
2. Check that `annotation` exists and is a list.
3. Collect every non-empty `sentence`.
4. Join multiple sentence fragments with one space.
5. Strip extra whitespace.
6. Skip the record if no transcript remains.

The script should not crash because of one bad JSON file.

It should count and report skipped reasons, such as:

- malformed JSON;
- missing audio;
- missing speaker id;
- empty sentence;
- duration too short;
- duration too long;
- text too long;
- speaker filtered out.

## 11. Audio Handling

The raw dataset audio is `.flac`.

CosyVoice tools use audio libraries such as `torchaudio` and `soundfile`. These may be able to read `.flac` directly if the environment has proper codec support.

Recommended first behavior:

```text
Keep wav.scp pointing directly to .flac files.
```

Reason:

- avoids unnecessary conversion;
- keeps preparation faster;
- preserves source audio;
- lets CosyVoice resample internally where needed.

If `.flac` fails on the server, then add or use optional conversion:

```text
.flac -> 24 kHz mono .wav
```

CosyVoice3 config uses:

```text
sample_rate: 24000
```

So converted audio should be:

```text
24 kHz, mono, WAV
```

## 12. Bengali Text Handling

Unlike VITS, CosyVoice3 does not require us to manually add Bengali characters to a small symbol table at the beginning.

CosyVoice3 uses a tokenizer-based pipeline tied to its pretrained model resources. The Bengali implementation should therefore:

- keep Bengali text as Unicode;
- avoid English-only cleaning rules;
- normalize whitespace;
- avoid destructive punctuation removal at first;
- keep Bangla punctuation when possible;
- let the CosyVoice3 tokenizer handle text as much as possible.

### Example Bengali Text Flow

Raw sentence inside JSON:

```json
{
  "annotation": [
    {
      "sentence": "তার কথাগুলো শুনে বুঝলাম  বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।"
    }
  ]
}
```

The preparation script reads:

```text
annotation[*]["sentence"]
```

Then it applies only light whitespace normalization.

Before normalization:

```text
তার কথাগুলো শুনে বুঝলাম  বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।
```

After normalization:

```text
তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।
```

The generated CosyVoice `text` file line becomes:

```text
01332512906_00d4da35-bebc-471b-aa92-0d9398388e98 তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।
```

Here:

```text
01332512906_00d4da35-bebc-471b-aa92-0d9398388e98 = utterance id
তার কথাগুলো ... গিয়েছে। = Bengali transcript
```

### What The Current Text Handling Does

The current `normalize_text()` function does:

```text
remove zero-width non-joiner \u200c
collapse repeated whitespace into one space
strip leading/trailing whitespace
```

Example:

```text
"  আমি   বাংলা ভাষায়  কথা বলি।  "
```

becomes:

```text
"আমি বাংলা ভাষায় কথা বলি।"
```

### What The Current Text Handling Does Not Do

The first baseline does not yet perform advanced Bengali normalization.

It does not yet convert:

```text
২০২৬ -> দুই হাজার ছাব্বিশ
১০:৩০ -> দশটা ত্রিশ
ড. -> ডাক্তার
kg -> কেজি
```

It also does not remove Bengali punctuation such as:

```text
। , ? !
```

This is intentional. The first goal is to keep Bengali text close to the original transcript and let CosyVoice3's tokenizer handle it.

Future Bengali improvements may include:

- Bengali number normalization;
- date/time normalization;
- abbreviation normalization;
- punctuation cleanup;
- optional Romanization experiments only if Bengali script fails badly;
- language tags or instruction prompts if needed.
## 13. Instruct Field For CosyVoice3

The official CosyVoice3 LibriTTS preparation adds an instruction string:

```text
You are a helpful assistant.<|endofprompt|>
```

The Bengali preparation should support an `--instruct` argument.

Initial safe default:

```text
You are a helpful assistant.<|endofprompt|>
```

Possible Bengali-specific future prompt:

```text
You are a helpful assistant. Please speak in Bengali.<|endofprompt|>
```

The exact instruction should be tested empirically. The first implementation should make it configurable.

## 14. Sequential Running Guide

The commands below describe the intended beginner sequence after implementation.

### Step 1: Clone Your Fork

```bash
git clone --recursive git@github.com:kawshikbuet17/cosyvoice-bengali-tts.git
cd cosyvoice-bengali-tts
git submodule update --init --recursive
```

### Step 2: Create Environment

```bash
conda create -n cosyvoice-bn python=3.10 -y
conda activate cosyvoice-bn
pip install -r requirements.txt
```

If `pip install -r requirements.txt` fails while building `openai-whisper==20231117` with this error:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

use the working server fix below, then rerun the requirements install:

```bash
conda activate cosyvoice-bn
cd ~/cosyvoice-bengali-tts

pip install "setuptools<81" wheel
python -c "import pkg_resources; print('pkg_resources ok')"

pip install openai-whisper==20231117 --no-build-isolation
pip install -r requirements.txt
```

Why this works:

```text
openai-whisper==20231117 expects pkg_resources during build.
Very new setuptools versions may not expose it in the temporary build environment.
Downgrading setuptools below 81 and installing Whisper without build isolation fixes that compatibility issue.
```

If GPU PyTorch needs to be installed separately, install the PyTorch build that matches the server CUDA/driver setup.

### Step 3: Download CosyVoice3 Pretrained Model

Using Hugging Face:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"
```

Using ModelScope:

```bash
python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')"
```

### Step 4: Test Original CosyVoice3 Inference

Run the original example on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python example.py
```

Inside Python, the selected physical GPU 1 will appear as `cuda:0`. This is normal because `CUDA_VISIBLE_DEVICES=1` hides the other GPUs from the process.

If this step or later staged scripts fail with `Permission denied`, `python3\r`, or `/usr/bin/env` errors after moving code from Windows to Linux, see **Section 15: Common Failure Points**. That section explains the `chmod +x`, `sed -i 's/\r$//' ...`, and `.gitattributes` fixes.

Expected result:

```text
one or more .wav files are generated
```

Do not continue to Bengali training until this works.

### Optional: Use The Staged Runner

Instead of running each command manually, you can use the staged runner that was added for this Bengali CosyVoice3 scaffold.

The staged runner is a convenience wrapper around the manual steps below. If you run `run.sh --stage 0 --stop_stage 5`, it automatically runs the Bengali data preparation, embedding extraction, speech token extraction, parquet creation, training list check, and training start.

Stage mapping:

```text
Stage 0 -> prepare Bengali metadata from raw .flac + .json
Stage 1 -> extract speaker embeddings
Stage 2 -> extract discrete speech tokens
Stage 3 -> create parquet files and data.list
Stage 4 -> check generated training lists
Stage 5 -> start CosyVoice3 training
```

So if you use the staged runner through Stage 5, you do not need to separately execute the later manual sections for preparing data, extracting embeddings, extracting speech tokens, making parquet files, or launching training. The manual sections are kept for debugging and for understanding what each stage does.

For long remote runs, first read **Section 16: Running Long Jobs With tmux**. That section explains how to keep the job alive after disconnecting from SSH or VS Code Remote, and how to make sure tmux uses the correct `cosyvoice-bn` Python environment.

The staged runner exports `CUDA_VISIBLE_DEVICES` before any stage starts. Therefore `--cuda_visible_devices 1` is intended to apply to speaker embedding extraction, speech token extraction, parquet generation, and training, not only Stage 5.

Important naming note:

- `run.sh` uses runner-style arguments such as `--speaker_id` and `--speaker_ids`;
- `prepare_bengali_data.py` uses Python-script arguments such as `--speaker-id` and `--speaker-ids`.

Both are correct in their own context.

From repo root:

```bash
cd examples/bengali/cosyvoice3
```

Single-speaker smoke test through all stages:

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --speaker_id 01332512906 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

Multi-speaker starter run, automatic speaker selection:

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --min_utterances_per_speaker 200 \
  --max_speakers 10 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

This automatically keeps up to 10 speakers that have at least 200 valid utterances after filtering.

Multi-speaker run for all eligible speakers:

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --min_utterances_per_speaker 200 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

This keeps every speaker that has at least 200 valid utterances after filtering. Use this after the single-speaker smoke test and the capped multi-speaker run are successful, because it may take much longer than the debug run.

Alternative multi-speaker run, explicit speaker IDs:

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --speaker_ids "01332512906 01700000001 01700000002" \
  --min_utterances_per_speaker 200 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

Use `--speaker_ids` when you already know exactly which speakers you want to train with. Put the speaker IDs inside quotes and separate them with spaces. The script still applies the duration, text length, and minimum utterance filters, so a listed speaker can still be skipped if it has too few valid utterances.

For safer debugging, run one stage at a time first:

```bash
bash run.sh --stage 0 --stop_stage 0 --dataset_root /home/kawshik/TTS_Dataset --speaker_id 01332512906
bash run.sh --stage 1 --stop_stage 1
bash run.sh --stage 2 --stop_stage 2
bash run.sh --stage 3 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
bash run.sh --stage 5 --stop_stage 5 --cuda_visible_devices 1
```

### Step 5: Prepare Bengali Data, Single-Speaker Smoke Test

> **Skip this manual step if you used the staged runner:** If you already ran `bash run.sh --stage 0 --stop_stage 5 ...` and it reached Stage 5, this step has already been handled by `run.sh`. Go directly to **Step 11: Monitor With TensorBoard**.


Command:

```bash
cd examples/bengali/cosyvoice3

python local/prepare_bengali_data.py \
  --dataset-root /home/kawshik/TTS_Dataset \
  --output-dir data \
  --speaker-id 01332512906 \
  --val-ratio 0.02 \
  --test-ratio 0.02 \
  --seed 1234 \
  --min-duration 0.5 \
  --max-duration 30.0 \
  --max-text-chars 250 \
  --instruct "You are a helpful assistant.<|endofprompt|>"
```

Expected output:

```text
data/bengali_train/wav.scp
data/bengali_train/text
data/bengali_train/utt2spk
data/bengali_train/spk2utt
data/bengali_dev/...
data/bengali_test/...
```

### Step 6: Prepare Bengali Data, Multi-Speaker

> **Skip this manual step if you used the staged runner:** If you already ran `bash run.sh --stage 0 --stop_stage 5 ...` and it reached Stage 5, this step has already been handled by `run.sh`. Go directly to **Step 11: Monitor With TensorBoard**.


Command:

```bash
python local/prepare_bengali_data.py \
  --dataset-root /home/kawshik/TTS_Dataset \
  --output-dir data \
  --min-utterances-per-speaker 200 \
  --max-speakers 10 \
  --val-ratio 0.02 \
  --test-ratio 0.02 \
  --seed 1234 \
  --min-duration 0.5 \
  --max-duration 30.0 \
  --max-text-chars 250 \
  --instruct "You are a helpful assistant.<|endofprompt|>"
```

### Step 7: Extract Speaker Embeddings

> **Skip this manual step if you used the staged runner:** If you already ran `bash run.sh --stage 0 --stop_stage 5 ...` and it reached Stage 5, this step has already been handled by `run.sh`. Go directly to **Step 11: Monitor With TensorBoard**.


Command from `examples/bengali/cosyvoice3`:

```bash
../../../tools/extract_embedding.py \
  --dir data/bengali_train \
  --onnx_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx

../../../tools/extract_embedding.py \
  --dir data/bengali_dev \
  --onnx_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx
```

Expected files:

```text
utt2embedding.pt
spk2embedding.pt
```

### Step 8: Extract Speech Tokens

> **Skip this manual step if you used the staged runner:** If you already ran `bash run.sh --stage 0 --stop_stage 5 ...` and it reached Stage 5, this step has already been handled by `run.sh`. Go directly to **Step 11: Monitor With TensorBoard**.


```bash
../../../tools/extract_speech_token.py \
  --dir data/bengali_train \
  --onnx_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx

../../../tools/extract_speech_token.py \
  --dir data/bengali_dev \
  --onnx_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx
```

Expected file:

```text
utt2speech_token.pt
```

### Step 9: Make Parquet Data

> **Skip this manual step if you used the staged runner:** If you already ran `bash run.sh --stage 0 --stop_stage 5 ...` and it reached Stage 5, this step has already been handled by `run.sh`. Go directly to **Step 11: Monitor With TensorBoard**.


```bash
mkdir -p data/bengali_train/parquet
../../../tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 10 \
  --src_dir data/bengali_train \
  --des_dir data/bengali_train/parquet

mkdir -p data/bengali_dev/parquet
../../../tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 10 \
  --src_dir data/bengali_dev \
  --des_dir data/bengali_dev/parquet
```

Expected files:

```text
data/bengali_train/parquet/data.list
data/bengali_dev/parquet/data.list
```

### Step 10: Start CosyVoice3 LLM Training

> **Skip this manual step if you used the staged runner:** If you already ran `bash run.sh --stage 0 --stop_stage 5 ...` and it reached Stage 5, this step has already been handled by `run.sh`. Go directly to **Step 11: Monitor With TensorBoard**.


For the first Bengali baseline, start with LLM fine-tuning.

```bash
export CUDA_VISIBLE_DEVICES=1

torchrun --nnodes=1 --nproc_per_node=1 \
  --rdzv_id=1986 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:1234 \
  ../../../cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config conf/cosyvoice3_bengali.yaml \
  --train_data data/bengali_train/parquet/data.list \
  --cv_data data/bengali_dev/parquet/data.list \
  --qwen_pretrain_path ../../../pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint ../../../pretrained_models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir exp/cosyvoice3_bengali/llm/torch_ddp \
  --tensorboard_dir tensorboard/cosyvoice3_bengali/llm/torch_ddp \
  --ddp.dist_backend nccl \
  --num_workers 2 \
  --prefetch 100 \
  --pin_memory \
  --use_amp
```

Training is considered successfully started when:

```text
loss logs appear
training steps increase
checkpoints are saved
TensorBoard logs are written
```


#### 10.1 Checkpoint Storage And Save Frequency

A real training run showed that CosyVoice3 LLM checkpoints are large:

```text
epoch_*_whole.pt ~= 1.9 GB each
```

Saving every epoch can quickly consume hundreds of GB. One run reached:

```text
data/        19G
exp/         200G
tensorboard/ 12M
```

and the root filesystem became full:

```text
/dev/sda5  2.7T total, 2.6T used, 371M available, 100% used
```

The training then failed while saving a checkpoint.

To reduce future storage usage, the Bengali config uses:

```yaml
save_interval_epochs: 10
```

and `cosyvoice/utils/executor.py` only saves whole checkpoints every `save_interval_epochs` epochs. With the current setting, future training should save roughly every 10 epochs instead of every epoch.

#### 10.2 Resume Training From A Checkpoint

The staged runner supports `--checkpoint`.

Example resume from epoch 104:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3

bash run.sh \
  --stage 5 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --min_utterances_per_speaker 200 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --checkpoint exp/cosyvoice3_bengali/llm/torch_ddp/epoch_104_whole.pt \
  --cuda_visible_devices 1
```

Use this only after enough disk space is available.

### Step 11: Monitor With TensorBoard

```bash
tensorboard --logdir examples/bengali/cosyvoice3/tensorboard --host 0.0.0.0 --port 6006
```

Then open this in a browser:

```text
http://SERVER_IP:6006
```

Replace `SERVER_IP` with the actual IP address or hostname of the training server.

If running on a remote server and the browser cannot connect, open the port through SSH tunneling or server access rules.

### Step 12: Inference After Training

A Bengali CosyVoice3 inference helper is included at:

```text
examples/bengali/cosyvoice3/local/infer_bengali.py
```

This script uses the original CosyVoice `AutoModel` API and supports:

- `zero_shot`
- `cross_lingual`
- `instruct2`

Important Bengali note:

```text
The script keeps text_frontend disabled by default.
```

Reason: the current CosyVoice frontend is mainly designed around Chinese/English normalization. For Bengali, the safer first baseline is to preserve Bangla Unicode text and pass it directly to the CosyVoice3 tokenizer.

#### 12.1 Current Verified Inference Status

The verified working inference path after the first Bengali LLM fine-tuning run is:

```text
fine-tuned epoch_104 checkpoint
  -> exported clean llm.pt
  -> complete CosyVoice3 model folder
  -> cross_lingual inference
  -> valid Bengali wav output
```

The current observation is:

```text
cross_lingual mode works well and generated a valid ~9 second Bengali output.
zero_shot mode is not yet the preferred path because it is sensitive to exact prompt-text alignment and produced a broken/0-second output during testing.
```

For now, use **cross-lingual mode** as the recommended Bengali inference path.

#### 12.2 Prepare A Clean Fine-Tuned Model Directory

CosyVoice inference expects a complete model directory containing files such as:

```text
cosyvoice3.yaml
llm.pt
flow.pt
hift.pt
campplus.onnx
speech_tokenizer_v3.onnx
CosyVoice-BlankEN/
```

Training checkpoints such as:

```text
examples/bengali/cosyvoice3/exp/cosyvoice3_bengali/llm/torch_ddp/epoch_104_whole.pt
```

are **training checkpoints**, not clean inference `llm.pt` files.

Meaning `epoch_104_whole.pt` contains:

```text
model weights
+ epoch
+ step
```

But `webui.py` and `AutoModel` expect `llm.pt` to contain only model weights. If `epoch` and `step` are still present, inference loading can fail with an error like:

```text
Unexpected key(s) in state_dict: "epoch", "step".
```

Create a clean inference model folder like this from repo root:

```bash
cd ~/cosyvoice-bengali-tts

rm -rf pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104
mkdir -p pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104

for f in pretrained_models/Fun-CosyVoice3-0.5B/*; do
  ln -s "$(realpath "$f")" "pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104/$(basename "$f")"
done

rm -f pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104/llm.pt
```

Then export a clean `llm.pt` by removing `epoch` and `step`:

```bash
python - <<'PY'
import torch
from pathlib import Path

src = Path("examples/bengali/cosyvoice3/exp/cosyvoice3_bengali/llm/torch_ddp/epoch_104_whole.pt")
dst = Path("pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104/llm.pt")

ckpt = torch.load(src, map_location="cpu")
if not isinstance(ckpt, dict):
    raise RuntimeError("Checkpoint is not a dict")
epoch = ckpt.pop("epoch", None)
step = ckpt.pop("step", None)
torch.save(ckpt, dst)
print("saved clean llm.pt")
print("removed epoch:", epoch)
print("removed step:", step)
print("output:", dst)
PY
```

Verify the clean file:

```bash
python - <<'PY'
import torch
path = "pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104/llm.pt"
ckpt = torch.load(path, map_location="cpu")
print("loaded clean llm.pt")
print("has epoch:", "epoch" in ckpt)
print("has step:", "step" in ckpt)
print("num keys:", len(ckpt))
PY
```

Expected:

```text
has epoch: False
has step: False
```

#### 12.3 Terminal Inference, Recommended Cross-Lingual Mode

Run from repo root:

```bash
cd ~/cosyvoice-bengali-tts
```

Use this as the current recommended inference command:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3/local/infer_bengali.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104 \
  --mode cross_lingual \
  --text "You are a Bengali text-to-speech assistant. Speak clearly and naturally.<|endofprompt|>আজ বিকেলে আকাশটা মেঘলা ছিল, কিন্তু ঠান্ডা বাতাসে হাঁটতে খুব ভালো লাগছিল।" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --output-dir inference_test/output_cross
```

Check the output:

```bash
ls -lh inference_test/output_cross
ffprobe -hide_banner inference_test/output_cross/*.wav
```

Expected result:

```text
A real wav file with non-zero duration.
During testing, cross_lingual generated a valid ~9 second output.
```

In cross-lingual mode:

```text
--text       = instruction + <|endofprompt|> + Bengali sentence to synthesize
--prompt-wav = reference voice/style audio
--prompt-text is not used
```

The prompt WAV can be Bengali or another language. The model uses it as the reference voice/style.

#### 12.4 Useful CosyVoice3 Instruction Prompts

The text before `<|endofprompt|>` is the CosyVoice3 instruction/system prompt.

Good options to try:

```text
You are a helpful assistant.<|endofprompt|>
```

```text
You are a Bengali text-to-speech assistant. Speak clearly and naturally.<|endofprompt|>
```

```text
You are a Bangla voice assistant. Read the text in natural Bengali pronunciation.<|endofprompt|>
```

```text
You are a professional Bengali narrator. Speak calmly and clearly.<|endofprompt|>
```

```text
You are a Bengali conversational speaker. Speak naturally, like everyday conversation.<|endofprompt|>
```

Example:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3/local/infer_bengali.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104 \
  --mode cross_lingual \
  --text "You are a professional Bengali narrator. Speak calmly and clearly.<|endofprompt|>আজকের সকালটা খুব শান্ত ছিল। নদীর পাশে হালকা বাতাস বইছিল।" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --output-dir inference_test/output_cross_narrator
```

#### 12.5 Zero-Shot Mode And Prompt Text

Zero-shot mode uses both:

```text
--prompt-wav
--prompt-text
```

Definitions:

```text
Prompt WAV  = short audio recording of the reference speaker
Prompt text = exact transcript of what is spoken inside that prompt WAV
```

Example:

```text
If ./asset/kawshik_prompt.wav says:
এটা একটা নমুনা ভয়েস। আমি আজকে খুব ভালো আছি।

Then --prompt-text should be:
এটা একটা নমুনা ভয়েস। আমি আজকে খুব ভালো আছি।
```

CosyVoice3 also needs a separator token:

```text
You are a helpful assistant.<|endofprompt|>
```

The updated `infer_bengali.py` automatically adds that prefix for CosyVoice3 if it is missing. A manual full command is:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3/local/infer_bengali.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104 \
  --mode zero_shot \
  --text "আজকের সকালটা খুব শান্ত ছিল। নদীর পাশে হালকা বাতাস বইছিল।" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --prompt-text "You are a helpful assistant.<|endofprompt|>এটা একটা নমুনা ভয়েস। আমি আজকে খুব ভালো আছি।" \
  --output-dir inference_test/output_zero_shot
```

Current finding:

```text
zero_shot reached the model but was not the best current path.
A broken/0-second output was observed during testing.
cross_lingual gave a valid output and is recommended for the current checkpoint.
```

#### 12.6 Gradio Web UI On Port 6007

The Gradio UI is useful for sharing an audio-output endpoint with others.

TensorBoard is for training curves:

```text
http://SERVER_IP:6006/
```

Gradio is for inferred audio output:

```text
http://SERVER_IP:6007/
```

Run the Gradio app from repo root:

```bash
cd ~/cosyvoice-bengali-tts

CUDA_VISIBLE_DEVICES=1 python webui.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104 \
  --port 6007
```

Open:

```text
http://202.4.122.62:6007/
```

The Bengali web UI includes:

- English and Bangla labels;
- short description of input/output;
- quick examples;
- prepared-by footer;
- prompt-audio upload;
- cross-lingual and zero-shot modes.

If browser audio upload fails, test terminal inference first. A Gradio upload failure can be caused by browser permissions, temporary disk issues, or server upload handling, while terminal inference can still work correctly.

#### 12.7 WebUI Troubleshooting And Audio Formats

**Audio Upload Permission Error (`/tmp/gradio`)**

If you see this error when uploading audio:

```text
PermissionError: [Errno 13] Permission denied: '/tmp/gradio/...'
```

Set a custom temporary directory before starting the webui:

```bash
export GRADIO_TEMP_DIR=/home/kawshik/gradio_tmp
mkdir -p /home/kawshik/gradio_tmp

CUDA_VISIBLE_DEVICES=1 python webui.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104 \
  --port 6007
```

**Audio Format Support**

The webui automatically handles any audio format:

- **WhatsApp voice notes** (OPUS) → auto-converted to 16kHz WAV
- **MP3, M4A, FLAC, OGG** → auto-converted to 16kHz WAV  
- **WAV files** → used directly if already 16kHz, otherwise resampled

Upload any audio file - the webui will resample it to the required 16 kHz automatically.

**WebUI Mode Instructions With `<|endofprompt|>`**

The webui now includes examples for each mode:

| Mode | Where `<|endofprompt|>` goes | Example |
|------|---------------------------|---------|
| **Zero-shot** | In **Prompt Text** field | `I am happy today.<\|endofprompt\|>` |
| **Cross-lingual** | In **TTS Text** field | `You are a helpful assistant.<\|endofprompt\|>আজকের আবহাওয়া খুব সুন্দর।` |
| **Instruct** | In **Instruction text** field | `Speak with a happy tone.<\|endofprompt\|>` |

The webui automatically adds `You are a helpful assistant.<|endofprompt|>` if you don't include it yourself.

#### 12.8 Command Syntax Rules

Use hyphen argument names, not underscore names:

```text
Correct:   --prompt-wav --prompt-text --model-dir --output-dir
Wrong:     --prompt_wav --prompt_text --model_dir --output_dir
```

For multi-line Bash commands, the backslash must be the final character on the line:

```bash
--prompt-wav ./asset/kawshik_prompt.wav \
```

Do not put a space after `\`:

```bash
--prompt-wav ./asset/kawshik_prompt.wav \ 
```

That breaks the command and can cause errors such as:

```text
--prompt_text: command not found
infer_bengali.py: error: the following arguments are required: --prompt-wav
```

#### 12.8 When To Use `--text-frontend`

The inference helper disables text frontend by default.

Use this default first:

```text
no --text-frontend flag
```

Only try `--text-frontend` as an experiment if you specifically want CosyVoice's built-in frontend normalization.

For the Bengali baseline, the recommended setting is still:

```text
text_frontend=False
```
## 15. Common Failure Points

### Missing submodules

Fix:

```bash
git submodule update --init --recursive
```

### Missing pretrained model files

Fix:

```text
download Fun-CosyVoice3-0.5B-2512 and check the local folder path
```

### Stage 1 says `Permission denied`

This can happen when `run.sh` calls CosyVoice tool scripts directly, but Linux does not have executable permission on those files.

Example error:

```text
run.sh: line 114: ../../../tools/extract_embedding.py: Permission denied
```

Fix from `examples/bengali/cosyvoice3`:

```bash
chmod +x ../../../tools/extract_embedding.py
chmod +x ../../../tools/extract_speech_token.py
chmod +x ../../../tools/make_parquet_list.py
```

Reason: `chmod +x` marks these Python files as executable scripts on Linux. Without this permission, Bash cannot run them directly even if Python itself is installed.

After fixing permission, resume from Stage 1. Stage 0 does not need to be rerun unless dataset filters changed.

```bash
bash run.sh \
  --stage 1 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --speaker_ids "01332512906" \
  --min_utterances_per_speaker 200 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

### Stage 1 says `python3\r` or `/usr/bin/env` cannot find Python

This can happen when code is edited or copied from Windows to Linux and script files keep Windows CRLF line endings.

Example error:

```text
/usr/bin/env: 'python3\r': No such file or directory
```

Reason: Linux expects line endings to be LF (`\n`). Windows often uses CRLF (`\r\n`). In an executable script, Linux reads the hidden `\r` at the end of the shebang line and tries to find a program named `python3\r` instead of `python3`.

Fix from `examples/bengali/cosyvoice3`:

```bash
sed -i 's/\r$//' ../../../tools/extract_embedding.py
sed -i 's/\r$//' ../../../tools/extract_speech_token.py
sed -i 's/\r$//' ../../../tools/make_parquet_list.py
sed -i 's/\r$//' run.sh
```

Then resume from Stage 1:

```bash
bash run.sh \
  --stage 1 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --speaker_ids "01332512906" \
  --min_utterances_per_speaker 200 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

### Prevent Windows-to-Linux line-ending issues with `.gitattributes`

This fork includes a repo-root `.gitattributes` file to keep script files Linux-friendly when the repository is moved between Windows and Linux.

Important entries:

```gitattributes
*.sh text eol=lf
*.py text eol=lf
*.md text eol=lf
*.json text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
```

After adding or changing `.gitattributes`, run this once from the repo root before committing:

```bash
git add --renormalize .
git status
```

This makes Git normalize tracked text files according to `.gitattributes`. The `.gitattributes` file prevents the problem in future checkouts, while `sed -i 's/\r$//' ...` fixes files that are already broken on the current Linux machine.

### tmux uses the wrong Python or asks for TorchCodec

This can happen if `tmux` starts with a polluted `PATH`. The prompt may show `(cosyvoice-bn)`, but `which python` may still point to `/opt/miniconda3/bin/python` or Python 3.13 instead of the actual `cosyvoice-bn` environment.

Common symptom:

```text
ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.
```

Check inside tmux:

```bash
which python
which python3
python --version
python3 --version
```

Correct paths should look like:

```text
/home/kawshik/miniconda3/envs/cosyvoice-bn/bin/python
/home/kawshik/miniconda3/envs/cosyvoice-bn/bin/python3
Python 3.10.x
Python 3.10.x
```

If the paths are wrong, run:

```bash
bash
source /home/kawshik/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice-bn
export CONDA_PREFIX=/home/kawshik/miniconda3/envs/cosyvoice-bn
export PATH="$CONDA_PREFIX/bin:/home/kawshik/miniconda3/condabin:/home/kawshik/miniconda3/bin:$PATH"
hash -r
```

Then verify `which python` and `which python3` again before running `run.sh`.
### Disk full while saving checkpoint

Observed error:

```text
RuntimeError: PytorchStreamWriter failed writing file data/291: file write failed
RuntimeError: unexpected pos ...
```

This means PyTorch failed while writing a checkpoint file. In the observed run, the actual cause was disk full, not a model bug.

Observed disk status:

```text
df -h .
/dev/sda5  2.7T total, 2.6T used, 371M available, 100% used
```

Inodes were fine:

```text
df -ih .
IUse% 5%
```

So the issue was storage capacity, not inode exhaustion.

Check storage:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3

df -h .
df -ih .
du -sh data exp tensorboard 2>/dev/null
find exp/cosyvoice3_bengali -type f -printf "%s %p\n" | sort -n | tail -20
```

Clean old checkpoints while keeping only epoch 100 and epoch 104:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3/exp/cosyvoice3_bengali/llm/torch_ddp

find . -maxdepth 1 -type f -name "epoch_*_whole.pt" \
  ! -name "epoch_100_whole.pt" \
  ! -name "epoch_104_whole.pt" \
  -print

find . -maxdepth 1 -type f -name "epoch_*_whole.yaml" \
  ! -name "epoch_100_whole.yaml" \
  ! -name "epoch_104_whole.yaml" \
  -print
```

If the printed list looks correct, delete them:

```bash
find . -maxdepth 1 -type f -name "epoch_*_whole.pt" \
  ! -name "epoch_100_whole.pt" \
  ! -name "epoch_104_whole.pt" \
  -delete

find . -maxdepth 1 -type f -name "epoch_*_whole.yaml" \
  ! -name "epoch_100_whole.yaml" \
  ! -name "epoch_104_whole.yaml" \
  -delete
```

Optional if you only want to keep the two selected checkpoints:

```bash
rm -f init.pt init.yaml
```

Deleting old checkpoint files does **not** delete TensorBoard graphs. TensorBoard event files live under `tensorboard/`, not inside the `.pt` checkpoint files.

### Training checkpoint is not a clean inference `llm.pt`

A file like this:

```text
epoch_104_whole.pt
```

is a training checkpoint. It may contain:

```text
model weights
+ epoch
+ step
```

A clean inference `llm.pt` should contain only model weights. If the checkpoint is copied directly to `llm.pt`, web UI or terminal inference can fail with:

```text
Unexpected key(s) in state_dict: "epoch", "step".
```

Fix: export a clean `llm.pt` as shown in **Step 12.2 Prepare A Clean Fine-Tuned Model Directory**.

### CosyVoice3 says `<|endofprompt|>` is missing

Observed error:

```text
AssertionError: <|endofprompt|> not detected in CosyVoice3 text or prompt_text
```

CosyVoice3 expects the special separator:

```text
<|endofprompt|>
```

For cross-lingual mode, put it inside `--text` before the Bengali target sentence:

```text
You are a Bengali text-to-speech assistant. Speak clearly and naturally.<|endofprompt|>বাংলা বাক্য...
```

For zero-shot mode, put it inside `--prompt-text` before the prompt transcript:

```text
You are a helpful assistant.<|endofprompt|>prompt wav-এর exact transcript...
```

The updated Bengali helper and web UI add this automatically for CosyVoice3 when possible, but writing it manually is also valid.

### Gradio upload fails but terminal inference works

The Gradio page may show an HTTP 500 error during prompt-audio upload. This can be caused by browser upload handling, temporary disk space, server file permissions, or a nearly full filesystem.

First prove the model works with terminal inference:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3/local/infer_bengali.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch104 \
  --mode cross_lingual \
  --text "You are a Bengali text-to-speech assistant. Speak clearly and naturally.<|endofprompt|>আজ বিকেলে আকাশটা মেঘলা ছিল, কিন্তু ঠান্ডা বাতাসে হাঁটতে খুব ভালো লাগছিল।" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --output-dir inference_test/output_cross
```

If this creates a valid wav, the model and checkpoint are fine. Then debug Gradio upload separately.

### Browser says no microphone found

This only affects browser recording. It does not mean inference is broken.

On a remote server, browser microphone access may fail because of HTTP, browser permissions, SSH/VS Code remote behavior, or missing client microphone forwarding.

Recommended path:

```text
Record prompt audio separately -> upload the wav file -> run inference
```

### Zero-shot generated 0-second or broken output

Zero-shot requires exact prompt alignment:

```text
prompt wav content == prompt text transcript
```

During testing, zero-shot reached the model but produced a broken/0-second output. Cross-lingual mode generated a valid ~9 second output and is the recommended current path for this checkpoint.
### `.flac` audio cannot be read

Possible fixes:

- install/fix audio backend dependencies;
- use optional conversion to 24 kHz mono WAV;
- verify `torchaudio` and `soundfile` can load the files.

### Too many bad JSON files

Check:

- JSON schema;
- `annotation` field;
- `sentence` field;
- speaker filtering.

### Training starts but quality is poor

This is normal early in development.

First validate:

- data preparation is correct;
- text matches audio;
- enough clean data exists;
- speaker selection is reasonable;
- training runs long enough.

### Confusing CosyVoice2 and CosyVoice3

Use CosyVoice3 paths consistently:

```text
examples/bengali/cosyvoice3
pretrained_models/Fun-CosyVoice3-0.5B
conf/cosyvoice3_bengali.yaml
```

## 16. Running Long Jobs With tmux

Long CosyVoice data preparation and training jobs should be run inside `tmux` on the remote server. If the SSH or VS Code remote connection closes, a normal terminal process may stop. A `tmux` session keeps running in the background.

Create a training session:

```bash
tmux new -s cosyvoice-train
```

Inside tmux, always initialize Conda and verify that the correct environment Python is being used:

```bash
bash
source /home/kawshik/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice-bn
export CONDA_PREFIX=/home/kawshik/miniconda3/envs/cosyvoice-bn
export PATH="$CONDA_PREFIX/bin:/home/kawshik/miniconda3/condabin:/home/kawshik/miniconda3/bin:$PATH"
hash -r

which python
which python3
python --version
python3 --version
```

Expected output:

```text
/home/kawshik/miniconda3/envs/cosyvoice-bn/bin/python
/home/kawshik/miniconda3/envs/cosyvoice-bn/bin/python3
Python 3.10.x
Python 3.10.x
```

If `python` or `python3` points to `/opt/miniconda3/bin/python` or shows Python 3.13, the wrong environment is being used. Do not start training until both commands point to the `cosyvoice-bn` environment.

Then run the staged runner from the Bengali CosyVoice3 folder:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3

bash run.sh \
  --stage 1 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset \
  --min_utterances_per_speaker 200 \
  --pretrained_model_dir ../../../pretrained_models/Fun-CosyVoice3-0.5B \
  --cuda_visible_devices 1
```

Detach from tmux without stopping the job:

```text
Ctrl+B, then D
```

List tmux sessions:

```bash
tmux ls
```

Reattach later:

```bash
tmux attach -t cosyvoice-train
```

Stop the running command inside tmux:

```text
Ctrl+C
```

Kill the whole tmux session only if you are sure you want to stop everything inside it:

```bash
tmux kill-session -t cosyvoice-train
```

For TensorBoard, use a second terminal or a second tmux session:

```bash
tmux new -s cosyvoice-tensorboard
source /home/kawshik/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice-bn
cd ~/cosyvoice-bengali-tts

tensorboard --logdir examples/bengali/cosyvoice3/tensorboard --host 0.0.0.0 --port 6006
```

Open:

```text
http://SERVER_IP:6006
```
## 17. What Must Be Configured On The Remote Server

Before running the full Bengali pipeline on the server, configure:

- dataset root, for example `/home/kawshik/TTS_Dataset`;
- pretrained model directory;
- GPU id through `CUDA_VISIBLE_DEVICES`;
- speaker selection for single-speaker or multi-speaker experiments;
- batch/training settings in `cosyvoice3_bengali.yaml`;
- number of data workers based on CPU/RAM;
- output directories for data, experiment logs, checkpoints, and TensorBoard.

## 18. Known Limitations

Current baseline limitations and observations:

- Bengali-specific text normalization is not yet advanced.
- Bengali phoneme/G2P support is not planned for the first version.
- `.flac` reading depends on the server audio backend.
- Full CosyVoice3 fine-tuning requires significant GPU memory, time, and storage.
- LLM checkpoints are large, about 1.9 GB each in the observed run.
- Saving every epoch is not practical for long runs; use `save_interval_epochs: 10` or another retention policy.
- Zero-shot inference is sensitive to exact prompt transcript alignment and did not perform as reliably in the first test.
- Cross-lingual inference is currently the verified working Bengali inference path.
- The current fine-tuned inference model uses the Bengali LLM checkpoint with the original pretrained flow and vocoder.

## 19. Beginner Glossary Of CosyVoice Keywords

This section explains the main keywords used in this guide in beginner-friendly language.

### Metadata

Metadata means information about the audio file, not the audio signal itself.

For this project, metadata includes:

- utterance id;
- audio path;
- transcript text;
- speaker id;
- duration;
- gender;
- train/dev/test split.

CosyVoice does not train directly from the raw JSON files. The raw JSON metadata must first be converted into CosyVoice metadata files such as `wav.scp`, `text`, `utt2spk`, and `spk2utt`.

### Utterance

An utterance means one spoken audio example.

In this dataset, one utterance usually means:

```text
one .flac file
one matching .json file
one transcript sentence or sentence group
```

### Utterance ID

An utterance ID is a unique name for one audio/text pair.

Example:

```text
01332512906_00d4da35-bebc-471b-aa92-0d9398388e98
```

CosyVoice uses utterance IDs to connect audio, text, speaker, embeddings, and speech tokens.

### Speaker ID

A speaker ID identifies whose voice is in the audio.

Example:

```text
01332512906
```

For single-speaker experiments, we keep only one `speaker_id`. For multi-speaker experiments, we keep multiple speaker IDs.

### `wav.scp`

`wav.scp` is a CosyVoice/Kaldi-style metadata file that maps utterance ID to audio file path.

Example:

```text
utt_000001 /home/kawshik/TTS_Dataset/Male/01332512906/sample.flac
```

Despite the name `wav.scp`, the path can point to `.flac` if the audio backend can read FLAC.

### `text`

`text` maps utterance ID to transcript.

Example:

```text
utt_000001 তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।
```

The Bengali preparation script extracts this text from `annotation[*]["sentence"]` in the raw JSON.

### `utt2spk`

`utt2spk` maps utterance ID to speaker ID.

Example:

```text
utt_000001 01332512906
```

This tells CosyVoice which speaker produced each utterance.

### `spk2utt`

`spk2utt` is the reverse of `utt2spk`. It maps one speaker ID to all utterances from that speaker.

Example:

```text
01332512906 utt_000001 utt_000002 utt_000003
```

CosyVoice tools use this to organize speaker-level information.

### Instruct

CosyVoice3 can use an instruction string to guide generation.

Default instruction used here:

```text
You are a helpful assistant.<|endofprompt|>
```

The Bengali preparation script writes this into an `instruct` file unless `--no-instruct` is used.

### Speaker Embedding

A speaker embedding is a numeric vector that represents a speaker's voice identity.

Beginner intuition:

```text
speaker embedding = compact voice fingerprint
```

It does not store the full audio. It stores features that help the model understand speaker similarity and speaker identity.

CosyVoice creates these files during embedding extraction:

```text
utt2embedding.pt
spk2embedding.pt
```

`utt2embedding.pt` stores one embedding per utterance. `spk2embedding.pt` stores one averaged embedding per speaker.

### Speech Token

A speech token is a discrete code extracted from speech audio.

Beginner intuition:

```text
speech token = compact symbolic representation of speech content/acoustics
```

CosyVoice uses speech tokens as part of the training target for its speech generation model.

The extraction step creates:

```text
utt2speech_token.pt
```

### Parquet

Parquet is a structured data storage format.

CosyVoice packs audio bytes, text, speaker information, embeddings, speech tokens, and optional instructions into parquet files before training.

Beginner intuition:

```text
parquet = packed training dataset file
```

Training reads `data.list`, and `data.list` points to the parquet files.

### `data.list`

`data.list` is a text file containing paths to parquet files.

Example:

```text
/path/to/parquet/parquet_000000000.tar
/path/to/parquet/parquet_000000001.tar
```

CosyVoice training uses this as the main training data input.

### LLM

In CosyVoice, LLM means the language-model-like part that predicts speech-related tokens from text and context.

For the first Bengali baseline, LLM fine-tuning is the safest first training target.

### Flow

Flow is the acoustic generation component that helps convert model representations into mel-like acoustic features.

For the first Bengali baseline, do not start by changing flow unless LLM training is already stable.

### HiFi-GAN / HIFT / Vocoder

The vocoder converts acoustic features into final waveform audio.

Beginner intuition:

```text
text/model features -> acoustic representation -> vocoder -> .wav audio
```

CosyVoice3 uses its vocoder-related components to produce the final sound you hear.

### Prompt Audio

Prompt audio is a short recording used to guide the speaker voice or style during inference.

For zero-shot or cross-lingual inference, prompt audio tells CosyVoice what kind of voice to imitate.

### Fine-Tuning

Fine-tuning means continuing training from a pretrained model using your own dataset.

For this project:

```text
pretrained CosyVoice3 + Bengali dataset -> Bengali-adapted checkpoint
```

### Checkpoint

A checkpoint is a saved model state during or after training.

Example:

```text
llm.pt
```

After training, checkpoints are used for inference or continued training.

### TensorBoard

TensorBoard is a web dashboard for monitoring training.

It can show:

- loss curves;
- learning rate;
- training progress;
- sometimes audio or other logged samples.

### Train / Dev / Test

The dataset is split into three parts:

- `train`: used to update the model;
- `dev`: used to monitor validation performance during training;
- `test`: kept aside for final checking.

For early debugging, `test` may be small, but it is still useful to keep it separate.
## 20. Next Steps

Recommended implementation order:

1. Run original CosyVoice3 inference.
2. Run Bengali preparation on a tiny subset.
3. Extract embeddings and speech tokens.
4. Generate parquet data.
5. Start LLM fine-tuning.
6. Document the first working inference result.
7. Improve Bengali normalization if needed.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

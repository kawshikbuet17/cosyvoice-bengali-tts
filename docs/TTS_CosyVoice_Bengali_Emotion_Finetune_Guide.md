# TTS CosyVoice Bengali Emotion Fine-Tune Guide

This guide explains how to fine-tune CosyVoice3 for **emotion-aware Bengali speech** using a Bengali dataset whose text files contain whole-line style tags such as `[warm]`, `[smile]`, `[pause]`, and `[angry]`.

This workflow is separate from the raw Bengali `.flac + .json` workflow described in [TTS_CosyVoice_Bengali_Implementation_Guide.md](TTS_CosyVoice_Bengali_Implementation_Guide.md).

Use this guide when:

- the dataset lives under `TTS_Dataset_Ghorer_Bazar`;
- each sample is stored as one `.txt` plus optional `_male.wav` and `_female.wav`;
- the text line may begin with emotion/style tags;
- the goal is to **fine-tune from an existing Bengali checkpoint** so the model learns more controllable expressive delivery.

For a beginner explanation of CosyVoice concepts such as `instruct`, `speaker embedding`, and speech tokens, see [TTS_CosyVoice_Beginner_Guide.md](TTS_CosyVoice_Beginner_Guide.md).

## 1. Purpose

The goal of this workflow is:

```text
emotion-tagged Bengali .txt + .wav dataset
  -> CosyVoice metadata with per-utterance instruct text
  -> speaker embeddings and speech tokens
  -> parquet training data
  -> resume Bengali LLM fine-tuning from an existing checkpoint
  -> Bengali expressive speech inference
```

This workflow does **not** replace the earlier Bengali scaffold. It adds a second Bengali fine-tuning path for a different dataset contract.

## 2. What CosyVoice3 Already Supports

CosyVoice3 already supports instruction-guided generation.

Examples already present in the repository include:

- dialect prompts
- speed prompts
- volume prompts
- emotion/style prompts

For example, CosyVoice3 already understands instruction-shaped inputs such as:

```text
Speak warmly and with a smile.<|endofprompt|>জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি।
```

So the core instruction mechanism already exists.

## 3. Why Fine-Tuning Is Still Needed

Even though CosyVoice3 already supports instruction-style prompting, fine-tuning is still needed here because:

- the Bengali checkpoint should learn this dataset's own delivery style more reliably;
- the dataset uses custom shorthand tags such as `[warm][smile]`, which are not native CosyVoice3 instruction strings;
- the earlier Bengali training flow did not contain explicit emotion labels, so emotion/style was only learned implicitly from audio;
- this workflow should make expressive Bengali delivery **more controllable**, not just occasionally emotional by chance.

So the main idea is:

```text
CosyVoice3 instruction mechanism already exists
  +
convert bracket tags into instruction text
  +
fine-tune on the tagged Bengali dataset
```

## 4. Dataset Contract

This workflow expects a dataset layout like:

```text
TTS_Dataset_Ghorer_Bazar/
  dataset/
    <uuid>.txt
    <uuid>_male.wav
    <uuid>_female.wav
```

Example:

```text
00527bc1-39f7-47fb-89e4-1c89a911904b.txt
00527bc1-39f7-47fb-89e4-1c89a911904b_male.wav
00527bc1-39f7-47fb-89e4-1c89a911904b_female.wav
```

Example text contents:

```text
[pause]অবশ্যই! অর্ডার নম্বরটা বলবেন, অথবা রেজিস্টার্ড ফোন নম্বরটা দিন।
```

```text
[warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।
```

What is used from this dataset:

- the `.txt` content;
- the leading whole-line emotion/style tags;
- `_male.wav` and `_female.wav`;
- the basename UUID used to pair text with audio.

How male and female wav files are handled:

- if both `_male.wav` and `_female.wav` exist for the same `.txt`, the prep script creates **two utterances**;
- both utterances share the same cleaned Bengali transcript and the same generated `instruct` string;
- the male wav is assigned to speaker ID `emotion_male` by default;
- the female wav is assigned to speaker ID `emotion_female` by default.

What is **not** used in the first implementation:

- per-sentence tag switching inside one line;
- nested or mid-sentence emotion tags;
- any `.json` metadata, because this dataset contract does not use `.json`.

## 5. How Tag Conversion Works

The first implementation supports **whole-line tags only**.

Supported input:

```text
[warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি।
```

Converted output:

```text
text:
জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি।

instruct:
You are a helpful assistant. Speak in natural Bengali. Speak warmly. Speak with a smile.<|endofprompt|>
```

This means:

- `[warm][smile]` is the dataset shorthand;
- CosyVoice3 training and inference still use instruction text with `<|endofprompt|>`;
- the new adapter layer converts one format into the other.

Initial supported tag mappings include:

- `[warm]` -> `Speak warmly.`
- `[smile]` -> `Speak with a smile.`
- `[pause]` -> `Pause briefly before speaking.`
- `[angry]` -> `Speak angrily.`
- `[sad]` -> `Speak sadly.`
- `[laugh]` / `[laughter]` -> `Include light laughter.`
- `[neutral]` -> `Speak neutrally.`

Some additional dataset-specific tags are also mapped, for example:

- `[clear]`
- `[serious]`
- `[gentle]`
- `[reassuring]`
- `[calming]`
- `[welcoming]`
- `[informative]`
- `[empathetic]`
- `[polite]`
- `[appreciative]`
- `[confidently]`

Unsupported tags are logged and ignored in v1.

## 6. Files Added For This Workflow

This workflow adds a separate Bengali emotion folder:

```text
examples/bengali/cosyvoice3_emotion/
  path.sh
  run.sh
  conf/
    cosyvoice3_bengali_emotion.yaml
    ds_stage2.json
  local/
    emotion_tags.py
    prepare_bengali_emotion_data.py
    infer_bengali_emotion.py
```

Purpose of each file:

- `emotion_tags.py`
  - parses leading tags;
  - maps tags to instruction fragments;
  - builds final CosyVoice3 `instruct` strings.

- `prepare_bengali_emotion_data.py`
  - reads the `.txt + _male.wav + _female.wav` dataset;
  - creates `wav.scp`, `text`, `utt2spk`, `spk2utt`, and `instruct`.

- `run.sh`
  - staged runner for prep, embeddings, speech tokens, parquet creation, and training.

- `infer_bengali_emotion.py`
  - inference helper that supports both natural instruction and bracket-tag shorthand.

## 7. Training Strategy

The first implementation keeps the training strategy simple and close to the earlier Bengali CosyVoice3 setup.

Main choices:

- fine-tune **LLM first**;
- reuse the Bengali CosyVoice3 configuration structure;
- resume from the earlier Bengali checkpoint path;
- keep checkpoint saving every 10 epochs;
- keep the old Bengali `.flac + .json` workflow untouched.

Recommended first checkpoint target:

```text
epoch_180_whole.pt
```

## 8. Stage Reference

The `run.sh` script uses numbered stages. Here is what each stage does:

| Stage | Name | Description |
|-------|------|-------------|
| **0** | Metadata preparation | Parses `.txt` files with emotion tags, creates `wav.scp`, `text`, `utt2spk`, `spk2utt`, and `instruct` files per split (train/dev/test) |
| **1** | Speaker embeddings | Extracts speaker embedding vectors from audio files using the pretrained speaker encoder |
| **2** | Speech tokens | Extracts speech tokens (neural codec representations) from audio files |
| **3** | Parquet creation | Converts all metadata, embeddings, and tokens into `.parquet` format for efficient training data loading |
| **4** | Data list check | Validates that all data lists are consistent and ready for training |
| **5** | Training | Runs the actual LLM fine-tuning (resumes from checkpoint if provided) |

**Typical workflows:**
- `--stage 0 --stop_stage 0`: Only prepare metadata (smoke test)
- `--stage 0 --stop_stage 5`: Full pipeline from metadata to training
- `--stage 5 --stop_stage 5`: Resume/continue training only (skips data prep)

## 9. Step 0: Go To The New Workflow Directory

From repo root on the remote server:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3_emotion
```

## 10. Step 1: Prepare Metadata Only

This is the safest first smoke test.

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 0 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar \
  --cuda_visible_devices 1
```

Expected output:

- `data/bengali_emotion_train`
- `data/bengali_emotion_dev`
- `data/bengali_emotion_test`
- `data/bengali_emotion_prep_report.json`
- `data/tag_stats.tsv`
- `data/unsupported_tag_stats.tsv`

Important generated files inside each split:

```text
wav.scp
text
utt2spk
spk2utt
instruct
```

## 11. Optional: Single-Speaker Smoke Test

You can keep only one gender speaker first:

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar \
  --speaker_id emotion_male \
  --checkpoint /home/kawshik/cosyvoice-bengali-tts/examples/bengali/cosyvoice3/exp/cosyvoice3_bengali/llm/torch_ddp/epoch_180_whole.pt \
  --cuda_visible_devices 1
```

or:

```bash
--speaker_id emotion_female
```

This is useful if you want a faster end-to-end validation before using both speakers together.

## 12. Main Multi-Speaker Fine-Tuning Command

Once the earlier Bengali checkpoint reaches epoch 180 and you want the emotion-aware fine-tune path, run:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3_emotion

bash run.sh \
  --stage 0 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar \
  --checkpoint /home/kawshik/cosyvoice-bengali-tts/examples/bengali/cosyvoice3/exp/cosyvoice3_bengali/llm/torch_ddp/epoch_180_whole.pt \
  --cuda_visible_devices 1
```

This does:

```text
1. parse text tags
2. create per-utterance instruct strings
3. extract speaker embeddings
4. extract speech tokens
5. build parquet files
6. resume LLM fine-tuning from epoch_180_whole.pt
```

The new outputs will go under:

```text
examples/bengali/cosyvoice3_emotion/exp/cosyvoice3_bengali_emotion/
examples/bengali/cosyvoice3_emotion/tensorboard/cosyvoice3_bengali_emotion/
```

> **Note:** If you ran the **Main Multi-Speaker Fine-Tuning Command** above (Section 12), you have already completed all stages (0-5). You can skip Section 13 (Manual Stage-by-Stage Commands) because the full pipeline already ran automatically.

## 13. Manual Stage-by-Stage Commands

If you want to run the stages separately:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3_emotion
```

Prepare metadata:

```bash
bash run.sh \
  --stage 0 \
  --stop_stage 0 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar
```

Extract speaker embeddings:

```bash
bash run.sh \
  --stage 1 \
  --stop_stage 1 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar
```

Extract speech tokens:

```bash
bash run.sh \
  --stage 2 \
  --stop_stage 2 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar
```

Build parquet:

```bash
bash run.sh \
  --stage 3 \
  --stop_stage 3 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar
```

Check lists:

```bash
bash run.sh \
  --stage 4 \
  --stop_stage 4 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar
```

Resume training:

```bash
bash run.sh \
  --stage 5 \
  --stop_stage 5 \
  --dataset_root /home/kawshik/TTS_Dataset_Ghorer_Bazar \
  --checkpoint /home/kawshik/cosyvoice-bengali-tts/examples/bengali/cosyvoice3/exp/cosyvoice3_bengali/llm/torch_ddp/epoch_180_whole.pt \
  --cuda_visible_devices 1
```

## 14. Direct Metadata Preparation Command

If you want to run the Python script directly instead of the staged runner:

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3_emotion

python local/prepare_bengali_emotion_data.py \
  --dataset-root /home/kawshik/TTS_Dataset_Ghorer_Bazar \
  --output-dir data \
  --prefix bengali_emotion \
  --male-speaker-id emotion_male \
  --female-speaker-id emotion_female \
  --base-instruct "You are a helpful assistant. Speak in natural Bengali." \
  --neutral-instruct "Speak neutrally."
```

## 15. Monitor With TensorBoard

From repo root on the server:

```bash
tensorboard --logdir examples/bengali/cosyvoice3_emotion/tensorboard --host 0.0.0.0 --port 6006
```

Then open:

```text
http://SERVER_IP:6006
```

The same remote `tmux`, `TensorBoard`, permission, and line-ending advice from the main Bengali implementation guide still applies here.

## 16. Inference After Training

The new helper supports both:

- natural instruction text;
- bracket-tag shorthand.

### 16.1 Cross-Lingual Tag Shorthand

This is the easiest first expressive inference smoke test because the earlier Bengali project already validated cross-lingual mode as the safer path.

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3_emotion/local/infer_bengali_emotion.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch180 \
  --mode cross_lingual \
  --text "[warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --output-dir inference_test/emotion_cross
```

This helper converts the leading tags into a CosyVoice3 instruction prefix before inference.

### 16.2 Instruct2 With Natural Instruction

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3_emotion/local/infer_bengali_emotion.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch180 \
  --mode instruct2 \
  --text "জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।" \
  --instruct-text "Speak warmly and with a smile.<|endofprompt|>" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --output-dir inference_test/emotion_instruct
```

### 16.3 Instruct2 With Tag Shorthand

```bash
CUDA_VISIBLE_DEVICES=1 python examples/bengali/cosyvoice3_emotion/local/infer_bengali_emotion.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch180 \
  --mode instruct2 \
  --text "[angry] অর্ডার নম্বর ছাড়া আমি এখনই অর্ডারটা খুঁজে দিতে পারছি না।" \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --output-dir inference_test/emotion_tags
```

The helper will:

```text
1. strip the leading tags from the transcript
2. convert tags into CosyVoice3 instruction text
3. run instruct2
```

### 16.4 Optional Gradio Demo

You can also use the main Gradio app for this workflow.

From repo root on the server:

```bash
cd ~/cosyvoice-bengali-tts

CUDA_VISIBLE_DEVICES=1 python webui.py \
  --model_dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch180 \
  --port 6007
```

Then open:

```text
http://SERVER_IP:6007
```

Recommended Gradio path for this workflow:

- use **Cross-lingual voice clone / ক্রস-লিঙ্গুয়াল ভয়েস ক্লোন** first;
- upload prompt audio;
- paste either:
  - a normal instruction-style input such as `You are a helpful assistant.<|endofprompt|>আজকের সকালটা খুব শান্ত ছিল।`; or
  - a tag-style input such as `[warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।`

Current Gradio behavior for this workflow:

- cross-lingual mode can now convert leading emotion tags into a CosyVoice instruction prefix automatically;
- instruct mode can derive instruction text automatically from leading tags when the instruction box is left empty;
- zero-shot is still available, but cross-lingual remains the safer first validation path.

## 17. What The New Prep Script Actually Writes

For each kept utterance, the new prep flow writes:

### `text`

Only the spoken Bengali sentence.

Example:

```text
emotion_male_00527bc1_39f7_47fb_89e4_1c89a911904b_male জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।
```

### `instruct`

The converted instruction string.

Example:

```text
emotion_male_00527bc1_39f7_47fb_89e4_1c89a911904b_male You are a helpful assistant. Speak in natural Bengali. Speak warmly. Speak with a smile.<|endofprompt|>
```

This is the key difference from the earlier Bengali workflow, which wrote the same generic `instruct` string for every utterance.

## 18. Current Limitations

The first implementation intentionally keeps the scope narrow:

- only **leading whole-line tags** are supported;
- `[warm][smile] sentence` is supported;
- `[emotion1]sentence1. [emotion2]sentence2.` is **not** supported in v1;
- unsupported tags are ignored and reported;
- the workflow uses existing CosyVoice3 instruction conditioning rather than introducing new tokenizer special tokens for dataset-specific tags.

## 19. Recommended First Validation Order

Use this order:

```text
1. Stage 0 only
2. Full runner on one speaker
3. Resume from epoch_180 on both speakers
4. Cross-lingual expressive inference
5. Instruct2 expressive inference
6. Optional Gradio validation
```

This keeps the debugging path small and easy to reason about.

## 20. Common Failure Points

For `tmux`, `TensorBoard`, and general permission issues, see Section 15 of the main Bengali implementation guide. Below are emotion-workflow-specific fixes.

### `run.sh: line X: $'\r': command not found`

This happens when `run.sh` (or other scripts) are edited or copied from Windows to Linux and keep Windows CRLF line endings.

Example error:

```text
run.sh: line 5: $'\r': command not found
run.sh: line 43: syntax error near unexpected token `$'in\r''
```

**Fix from `examples/bengali/cosyvoice3_emotion`:**

```bash
cd ~/cosyvoice-bengali-tts/examples/bengali/cosyvoice3_emotion
sed -i 's/\r$//' run.sh
```

**Reason:** Linux expects line endings to be LF (`\n`). Windows often uses CRLF (`\r\n`). The hidden `\r` characters cause Bash to fail parsing the script.

After fixing line endings, rerun your command. You do not need to rerun earlier stages unless the actual script logic changed.

### Prevent Windows-to-Linux line-ending issues with `.gitattributes`

The repo-root `.gitattributes` file helps keep script files Linux-friendly when the repository is moved between Windows and Linux.

If you still encounter line-ending errors after a fresh clone or checkout, verify Git is applying the settings:

```bash
cd ~/cosyvoice-bengali-tts
git add --renormalize .
git status
```

This makes Git normalize tracked text files according to `.gitattributes` rules. The `.gitattributes` file prevents the problem in future checkouts, while `sed -i 's/\r$//'` fixes files that are already broken on the current Linux machine.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

# Major Changes From Original CosyVoice

This document gives a high-level summary of how this fork differs from the original `FunAudioLLM/CosyVoice` repository.

The purpose of the fork is simple: keep the original CosyVoice3 model and training flow mostly intact, but add a clear Bengali TTS adaptation path for a raw `.flac` + `.json` speech dataset.

Current status:

```text
The Bengali CosyVoice3 scaffold has been added and executed on the remote dataset.
The pipeline can prepare Bengali metadata, extract embeddings/tokens, build parquet files, and start LLM fine-tuning.
A fine-tuned epoch_104 checkpoint was exported into a clean inference `llm.pt`.
Cross-lingual Bengali inference is verified and generated a valid non-empty wav output.
Zero-shot inference is documented but is not the recommended first inference path yet because it is sensitive to exact prompt transcript alignment.
```

## 1. What This Fork Adds

Original CosyVoice already provides strong multilingual TTS, inference, training examples, and deployment tools.

This fork adds a Bengali adaptation direction:

```text
raw Bengali .flac + .json dataset
  -> CosyVoice metadata
  -> embeddings and speech tokens
  -> parquet data
  -> CosyVoice3 fine-tuning
  -> Bengali speech inference
```

The main target is:

```text
CosyVoice3-based Bengali TTS
```

## 2. Dataset Support Direction

The original CosyVoice examples use prepared datasets such as LibriTTS.

Those examples expect data to be converted into:

```text
wav.scp
text
utt2spk
spk2utt
```

and then into:

```text
utt2embedding.pt
spk2embedding.pt
utt2speech_token.pt
parquet files
data.list
```

The Bengali dataset starts differently:

```text
TTS_Dataset/
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

Text is extracted from:

```text
annotation[*]["sentence"]
```

The Bengali data bridge converts this raw dataset into the CosyVoice metadata format.
At high level, the current implementation mainly uses the transcript under `annotation[*]["sentence"]`, speaker identity, audio-path linkage, and optional `duration` / `speech_id` fields. Word-level timing data is kept in the schema example but is not yet used as a primary training signal.

## 3. CosyVoice3 First

This fork will focus on CosyVoice3 first.

Reasons:

- CosyVoice3 is the newest model line in the repository.
- The original README recommends Fun-CosyVoice3 for stronger performance.
- CosyVoice3 has an official training example under `examples/libritts/cosyvoice3`.
- It is the best fit for Bengali multilingual/cross-lingual experimentation.

CosyVoice2 may be used as a reference, but it is not the first target.

## 4. Bengali Workflow

The workflow is:

```text
1. Run original CosyVoice3 inference.
2. Prepare Bengali raw data into CosyVoice metadata.
3. Extract speaker embeddings.
4. Extract speech tokens.
5. Generate parquet data.
6. Start CosyVoice3 LLM fine-tuning.
7. Manage large checkpoints and export a clean inference `llm.pt`.
8. Test Bengali inference, currently verified through cross-lingual mode.
9. Expand from a small subset to larger single-speaker or multi-speaker training.
```

This keeps the implementation close to original CosyVoice and avoids unnecessary refactoring.

## 5. Single-Speaker And Multi-Speaker Direction

The Bengali implementation should support both paths.

Single-speaker use:

```text
select one speaker_id
debug the full pipeline
prove training and inference can run
```

Multi-speaker use:

```text
keep multiple speakers
use utt2spk/spk2utt metadata
scale toward a richer Bengali TTS model
```

CosyVoice already uses speaker-aware metadata, so multi-speaker support should be added through dataset preparation and documentation first, not by redesigning the core model.

## 6. Audio Handling Direction

The Bengali dataset uses `.flac`.

The first implementation should try to keep `.flac` paths directly in `wav.scp`, because CosyVoice audio tools may read them through `torchaudio` or `soundfile`.

If `.flac` loading fails on the server, the preparation script should support optional conversion:

```text
.flac -> 24 kHz mono .wav
```

CosyVoice3 uses:

```text
sample_rate: 24000
```

## 7. Bengali Text Handling Direction

Unlike the earlier VITS adaptation, CosyVoice3 does not require a manually edited Bengali character symbol table for the first baseline.

The Bengali text path is:

- keep Bengali text as Unicode;
- normalize whitespace;
- avoid English-only cleaning;
- preserve useful Bengali punctuation;
- let the CosyVoice3 tokenizer handle the text first;
- add Bengali number/date normalization later if needed.

## 8. Documentation Added

Current Bengali-focused docs and examples:

```text
docs/TTS_CosyVoice_Beginner_Guide.md
docs/TTS_CosyVoice_Bengali_Implementation_Guide.md
docs/Major_Changes_From_Original_CosyVoice.md
docs/sample_bengali_raw_dataset/
.gitattributes
```

The beginner guide explains TTS and CosyVoice concepts.

The Bengali implementation guide explains the Bengali dataset adaptation and sequential running path.

This file summarizes the high-level differences from original CosyVoice.

A root `.gitattributes` file was also added to keep `.sh` and `.py` files with Linux LF line endings. This prevents Windows-to-Linux script errors such as `python3\r` when running executable Python tools on a server.

## 9. What Should Stay Close To Original CosyVoice

The fork should preserve the original project structure as much as possible:

- keep `cosyvoice/` model code mostly unchanged;
- reuse `tools/extract_embedding.py`;
- reuse `tools/extract_speech_token.py`;
- reuse `tools/make_parquet_list.py`;
- reuse `cosyvoice/bin/train.py`;
- base Bengali training on `examples/libritts/cosyvoice3`;
- add Bengali code under examples/bengali/cosyvoice3;
- include a Bengali inference helper for zero-shot, cross-lingual, and instruct-style testing.

The Bengali changes should be a focused adaptation layer, not a broad rewrite.

## 10. Next Steps

Recommended next actions:

1. Keep cross-lingual inference as the verified Bengali audio-output path.
2. Resume training from the latest good checkpoint after freeing disk space.
3. Save checkpoints less frequently, for example every 10 epochs.
4. Continue testing Gradio on port `6007` for manager/user audio review.
5. Debug zero-shot separately after prompt transcript alignment is controlled.
6. Improve Bengali normalization and scale training after stable inference is confirmed.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

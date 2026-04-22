# TTS and CosyVoice for Beginners: A Practical Starting Guide

This guide explains what CosyVoice is, how it fits into Text-to-Speech work, and how a beginner can approach this repository before making Bengali dataset-specific changes.

For this project, the main direction is **CosyVoice3-based Bengali TTS adaptation**. The input will eventually be Bengali text with matching `.flac` audio and `.json` metadata, and the output will be a trained or adapted model that can synthesize Bengali speech as audio.

---

## Table of Contents

1. [What is TTS?](#1-what-is-tts)
2. [What is CosyVoice?](#2-what-is-cosyvoice)
3. [Why CosyVoice3?](#3-why-cosyvoice3)
4. [How CosyVoice is different from VITS](#4-how-cosyvoice-is-different-from-vits)
5. [Main CosyVoice concepts](#5-main-cosyvoice-concepts)
6. [What the LLM does in CosyVoice3](#6-what-the-llm-does-in-cosyvoice3)
7. [CosyVoice3 architecture at a glance](#7-cosyvoice3-architecture-at-a-glance)
8. [What the original repo provides](#8-what-the-original-repo-provides)
9. [Important folders in this repo](#9-important-folders-in-this-repo)
10. [The beginner mental model](#10-the-beginner-mental-model)
11. [Inference vs training](#11-inference-vs-training)
12. [CosyVoice3 data pipeline](#12-cosyvoice3-data-pipeline)
13. [Dataset format CosyVoice expects](#13-dataset-format-cosyvoice-expects)
14. [How Bengali data will fit later](#14-how-bengali-data-will-fit-later)
15. [Environment setup overview](#15-environment-setup-overview)
16. [Pretrained model download overview](#16-pretrained-model-download-overview)
17. [First thing to run](#17-first-thing-to-run)
18. [Training roadmap](#18-training-roadmap)
19. [Common beginner mistakes](#19-common-beginner-mistakes)
20. [Glossary](#20-glossary)
21. [Final summary](#21-final-summary)

---

## 1. What is TTS?

**TTS** means **Text-to-Speech**.

The basic task is:

```text
Input:  text
Output: spoken audio
```

Example:

```text
Input text:
Hello, how are you?

Output audio:
hello_how_are_you.wav
```

A TTS model must learn pronunciation, timing, pauses, rhythm, pitch, speaker style, and final waveform generation. That is why TTS is harder than a simple text or audio classification task.

---

## 2. What is CosyVoice?

**CosyVoice** is a modern open-source Text-to-Speech system from FunAudioLLM.

It supports:

- zero-shot voice cloning
- cross-lingual speech generation
- instruction-controlled speech generation
- training and fine-tuning workflows
- deployment examples

In simple words:

```text
CosyVoice can generate speech from text, and it can also use prompt audio to imitate a speaker voice.
```

The repository contains multiple model generations:

- CosyVoice 1.0
- CosyVoice 2.0
- Fun-CosyVoice 3.0

For this Bengali project, we will focus on **CosyVoice3** first.

---

## 3. Why CosyVoice3?

CosyVoice3 is the newest line in this repository.

It is designed for stronger:

- multilingual generation
- cross-lingual generation
- speaker similarity
- prosody naturalness
- instruction following

That makes it a better starting point for Bengali TTS adaptation than older CosyVoice versions.

Important note:

CosyVoice3 has strong multilingual ability, but Bengali support still needs practical validation and adaptation. This project will prepare the Bengali dataset and training/inference workflow so we can test and improve Bengali speech generation.

---

## 4. How CosyVoice is different from VITS

VITS and CosyVoice are both TTS systems, but they are very different in workflow.

### VITS-style beginner view

VITS commonly uses a simple filelist:

```text
audio_path|text
```

Then training reads audio and text directly from that filelist.

### CosyVoice-style beginner view

CosyVoice uses a richer staged pipeline:

```text
Raw audio and text
  -> wav.scp / text / utt2spk / spk2utt
  -> speaker embeddings
  -> speech tokens
  -> parquet files
  -> model training
```

So for CosyVoice, dataset preparation is more involved than VITS.

The good news: most of the heavy feature extraction and parquet generation already exists in the original CosyVoice repo. Our Bengali work should reuse those tools instead of rewriting them.

---

## 5. Main CosyVoice concepts

### Text

The sentence that should be spoken.

For this project, the text will later be Bengali.

### Prompt audio

A short audio recording used to tell the model what speaker voice or style to imitate.

### Speaker embedding

A compact vector representation of a speaker's voice.

CosyVoice uses speaker embeddings to help model speaker identity.

### Speech token

A discrete representation extracted from speech audio.

CosyVoice uses speech tokens as part of its speech modeling pipeline.

### Parquet data

CosyVoice packs prepared training examples into `.parquet` files.

This is different from VITS, which can train from plain text filelists.

---

## 6. What the LLM does in CosyVoice3

In CosyVoice3, **LLM** means the language-model-like part of the TTS system. It is not used as a chatbot. Its job is to understand the text and context, then predict speech-related tokens that the rest of the speech system can turn into audio.

So yes, in this Bengali project we are training and inferencing an LLM, but it is important to say this precisely:

```text
Correct:
We fine-tune the CosyVoice3 LLM component for TTS.

Not correct:
We train a standalone chatbot-style LLM that directly outputs audio.
```

The LLM is only one component inside the full CosyVoice3 speech system. The final audio still needs the flow/acoustic generator, vocoder, tokenizer resources, and model assets from the pretrained CosyVoice3 folder.

Beginner intuition:

```text
The LLM is the planner.

It decides:
- what should be spoken
- roughly how the spoken content should unfold over time
- which speech-token sequence should represent that spoken content
```

### Why CosyVoice3 uses an LLM

Older TTS systems often learn a more direct mapping:

```text
text -> acoustic features -> waveform
```

CosyVoice3 uses a richer representation:

```text
text + prompt/context -> LLM -> speech tokens -> acoustic model -> waveform
```

This helps CosyVoice3 handle multilingual text, cross-lingual voice cloning, longer context, and more natural prosody. For Bengali adaptation, the LLM is especially important because it learns how Bengali text should map into the speech-token patterns that later components can synthesize.

### How the LLM works during inference

Inference means the model is generating speech without changing its weights.

At inference time, the LLM works like this:

1. Bengali input text is converted into text tokens.
2. Prompt audio and speaker information provide voice/style context.
3. The LLM reads the text tokens plus context.
4. The LLM predicts speech tokens step by step.
5. The predicted speech tokens are passed to the acoustic generator and vocoder.
6. The system writes the final waveform audio.

Diagram:

```text
Input Bengali text
      |
      v
Text tokenizer
      |
      v
Text tokens
      |
      +----------------------------+
      |                            |
      v                            v
Prompt audio                 Speaker embedding
      |                            |
      v                            |
Prompt / speech context -----------+
      |
      v
+------------------------------+
| CosyVoice3 LLM               |
| predicts speech tokens       |
+------------------------------+
      |
      v
Predicted speech tokens
      |
      v
Flow / acoustic generator
      |
      v
Mel-like acoustic features
      |
      v
Vocoder / HiFT / HiFi-GAN
      |
      v
Final .wav speech audio
```

The important point is that the LLM does not directly output `.wav` audio. It predicts an intermediate speech representation. The flow model and vocoder are still needed to create the final sound.

During inference, the LLM weights stay fixed. The model is only using what it has already learned to produce new speech-token sequences from new text.

In practice, inference uses the full CosyVoice3 model folder. If Bengali fine-tuning produces a new `llm.pt`, that file replaces the LLM checkpoint inside a complete model folder. It does not replace the whole TTS system by itself.

### How the LLM works during training

Training means the model is learning from known examples and updating its weights.

At training time, the LLM works like this:

1. Each training example has text and matching real audio.
2. The text is converted into text tokens.
3. The real audio is converted into target speech tokens by the speech tokenizer.
4. The LLM tries to predict the speech tokens from the text and context.
5. The prediction is compared with the target speech tokens.
6. The difference becomes the loss.
7. Backpropagation updates the LLM weights so the next prediction becomes better.

Diagram:

```text
Training text ----------------------+
                                    |
                                    v
                              Text tokenizer
                                    |
                                    v
                              Text tokens
                                    |
                                    v
Real audio -> speech tokenizer -> target speech tokens
                                    |
                                    v
                         +--------------------+
                         | CosyVoice3 LLM     |
                         | predicts tokens    |
                         +--------------------+
                                    |
                                    v
                         Compare prediction with
                         target speech tokens
                                    |
                                    v
                         Update LLM weights
```

That is why the data preparation step creates `utt2speech_token.pt`. Those speech tokens become the training target for the LLM.

During training, the LLM is not only producing tokens; it is being corrected. If it predicts the wrong speech-token sequence, the loss becomes higher. If it predicts a better sequence, the loss becomes lower and accuracy usually improves. This is why a healthy TensorBoard curve often shows `TRAIN/loss` going down and `TRAIN/acc` going up.

For the current Bengali baseline, the staged training command focuses on the LLM model. Conceptually, it fine-tunes:

```text
pretrained CosyVoice3 llm.pt
  -> Bengali training examples
  -> adapted Bengali llm.pt
```

That adapted `llm.pt` is then used together with the other pretrained CosyVoice3 components for inference.

### What the LLM learns

The LLM learns several useful things:

- how text tokens connect to speech-token sequences
- how long or short the spoken output should be
- how punctuation and sentence structure affect pauses and rhythm
- how prompt context and speaker information should influence generation
- how to keep content consistent so the generated speech says the requested text

For Bengali, LLM fine-tuning should help the model become better at Bengali text-to-speech alignment. It can improve pronunciation patterns, timing, and content consistency, but it does not solve every audio-quality problem alone. If the LLM predicts reasonable speech tokens but the final sound is still poor, the issue may be in data quality, prompt selection, the flow model, the vocoder, or insufficient training.

### Why this project starts with LLM fine-tuning

Starting with LLM fine-tuning is a practical first step because it targets the text-to-speech-token mapping without immediately changing every part of the system.

For this Bengali project:

```text
First target:
fine-tune the LLM so Bengali text maps better to speech tokens.

Later targets if needed:
flow fine-tuning, vocoder-related work, stronger normalization, larger data.
```

So when TensorBoard shows `TRAIN/loss`, `TRAIN/acc`, and LLM checkpoints, we are mainly watching whether the LLM is learning this text-to-speech-token prediction task.

---

## 7. CosyVoice3 architecture at a glance

CosyVoice3 is not one single model block. It is a speech generation system made of several cooperating parts.

Beginner view:

```text
CosyVoice3 = tokenizer + LLM + flow/acoustic generator + vocoder + support resources
```

The easiest way to understand it is to separate **what plans the speech** from **what makes the sound**.

### Main architecture blocks

```text
Text tokenizer
  Converts input text into text tokens.

Speaker embedding / prompt context
  Represents speaker identity and prompt-audio style.

CosyVoice3 LLM
  Predicts speech tokens from text tokens and context.

Flow / acoustic generator
  Converts predicted speech tokens into acoustic features.

Vocoder / HiFT / HiFi-GAN
  Converts acoustic features into final waveform audio.

ONNX and tokenizer resources
  Support feature extraction, speech-token extraction, and model input processing.
```

### Full inference architecture

During inference, CosyVoice3 starts from text and optional prompt audio, then produces a `.wav` file:

```text
                +----------------------+
                |  Bengali input text  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   Text tokenizer     |
                +----------+-----------+
                           |
                           v
                     Text tokens
                           |
                           v
Prompt audio ---> Speaker / prompt context
                           |
                           v
                +----------------------+
                |    CosyVoice3 LLM    |
                | predicts speech      |
                | tokens               |
                +----------+-----------+
                           |
                           v
                  Predicted speech tokens
                           |
                           v
                +----------------------+
                | Flow / acoustic      |
                | generator            |
                +----------+-----------+
                           |
                           v
                 Acoustic representation
                           |
                           v
                +----------------------+
                | Vocoder / HiFT       |
                +----------+-----------+
                           |
                           v
                     Final .wav audio
```

So the LLM is central, but it is not the whole system. It predicts speech tokens. The flow and vocoder turn those speech tokens into actual sound.

### Full training architecture

During training, the system uses real audio to create targets for the model:

```text
Training text ----------------------+
                                    |
                                    v
                              Text tokenizer
                                    |
                                    v
                              Text tokens
                                    |
                                    v
Real audio -> speech tokenizer -> target speech tokens
                                    |
                                    v
                             CosyVoice3 LLM
                                    |
                                    v
                         predicted speech tokens
                                    |
                                    v
                         compare with target tokens
                                    |
                                    v
                              update weights
```

For the first Bengali baseline, the main weight update is for the LLM part. The flow and vocoder are still part of inference, but they are not the first training target.

### What the pretrained folder contains

A complete CosyVoice3 pretrained model folder usually contains several pieces:

- `llm.pt`
- flow checkpoint
- vocoder / HiFT checkpoint
- speech tokenizer ONNX file
- speaker embedding ONNX file
- tokenizer or Qwen-related resources
- config files

This is why a fine-tuned Bengali `llm.pt` alone is not enough for inference. It must be used inside a complete CosyVoice3 model folder with the other components.

### What changes in this Bengali baseline

For the current Bengali path:

```text
Changed by training:
CosyVoice3 LLM checkpoint

Reused from pretrained model:
flow, vocoder, ONNX resources, tokenizer resources, config structure
```

This makes the first experiment smaller and easier to validate. If Bengali output still needs improvement after LLM fine-tuning, later experiments may consider flow fine-tuning, better normalization, more data, or prompt selection.

---

## 8. What the original repo provides

The original CosyVoice repo already provides:

- pretrained model usage examples
- CosyVoice3 inference example
- LibriTTS training example
- embedding extraction tool
- speech token extraction tool
- parquet generation tool
- model training entrypoint
- web demo
- runtime and deployment examples

For Bengali adaptation, we should not rewrite these systems first.

The first goal should be:

```text
Convert Bengali raw data into the format CosyVoice already understands.
```

---

## 9. Important folders in this repo

### `cosyvoice/`

Main Python package for model code, inference, dataset processing, tokenization, flow, LLM, and vocoder components.

### `tools/`

Utility scripts used during data preparation.

Important tools include:

- `extract_embedding.py`
- `extract_speech_token.py`
- `make_parquet_list.py`

### `examples/libritts/cosyvoice3/`

Official CosyVoice3 training example using LibriTTS-style data.

This is the main reference for the Bengali training pipeline.

### `example.py`

Basic inference examples for CosyVoice, CosyVoice2, and CosyVoice3.

### `pretrained_models/`

This folder is not included by default. It is where downloaded pretrained CosyVoice models are usually placed.

---

## 10. The beginner mental model

Think of CosyVoice3 work in three layers.

### Layer 1: Inference

Use a downloaded pretrained model to generate audio.

This confirms that the environment and model files work.

### Layer 2: Dataset preparation

Convert your raw dataset into CosyVoice training format.

This is where most Bengali-specific work will happen.

### Layer 3: Training or fine-tuning

Use prepared parquet data to fine-tune parts of CosyVoice3.

Start small before attempting a full training pipeline.

---

## 11. Inference vs training

### Inference

Inference means:

```text
Use an existing model to generate speech.
```

You do not change model weights during inference.

### Training

Training means:

```text
Update model weights using a dataset.
```

Training needs:

- GPU
- prepared dataset
- config file
- pretrained checkpoint or initialization
- time
- monitoring

For this project, we should run pretrained inference first, then Bengali data preparation, then training.

---

## 12. CosyVoice3 data pipeline

The official CosyVoice3 example follows this general flow:

```text
Step 1: Prepare metadata
        wav.scp, text, utt2spk, spk2utt

Step 2: Extract speaker embeddings
        utt2embedding.pt, spk2embedding.pt

Step 3: Extract speech tokens
        utt2speech_token.pt

Step 4: Generate parquet files
        parquet files and data.list

Step 5: Train model
        usually start with llm training
```

This is the structure we should preserve for Bengali support.

---

## 13. Dataset format CosyVoice expects

CosyVoice examples use a Kaldi-style metadata format.

### `wav.scp`

Maps utterance id to audio path.

```text
utt_000001 /path/to/audio_000001.wav
utt_000002 /path/to/audio_000002.wav
```

For Bengali, these paths may point to `.flac` files first if the audio backend can read them.

### `text`

Maps utterance id to transcript.

```text
utt_000001 This is the text for the first utterance.
utt_000002 This is the text for the second utterance.
```

For Bengali:

```text
utt_000001 তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।
```

The actual Bengali implementation guide will show Bangla-script examples.

### `utt2spk`

Maps utterance id to speaker id.

```text
utt_000001 speaker_001
utt_000002 speaker_001
```

### `spk2utt`

Maps speaker id to all utterances from that speaker.

```text
speaker_001 utt_000001 utt_000002
```

---

## 14. How Bengali data will fit later

The Bengali raw dataset is expected to look like:

```text
TTS_Dataset/
  Male/
    speaker_id/
      optional_nested_folders/
        audio.flac
        audio.json
  Female/
    speaker_id/
      optional_nested_folders/
        audio.flac
        audio.json
```

The transcript is expected inside JSON:

```text
annotation[*]["sentence"]
```

The Bengali preparation script should:

- recursively scan the dataset
- pair `.flac` and `.json`
- extract Bengali transcript text
- keep one speaker for single-speaker experiments if requested
- keep multiple speakers for multi-speaker experiments if requested
- create train/dev/test splits
- write CosyVoice metadata files
- write a report showing kept and skipped files

The project-specific implementation steps are documented separately in:

```text
docs/TTS_CosyVoice_Bengali_Implementation_Guide.md
```

---

## 15. Environment setup overview

The original repo recommends Python 3.10.

For beginners, the important idea is that CosyVoice needs a clean, dedicated Python environment. Do not mix it with unrelated projects or system Python packages.

### `openai-whisper` / `pkg_resources` install issue

On some servers, dependency installation may fail around `openai-whisper` and `pkg_resources`.

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

This is a dependency build compatibility issue, not a Bengali dataset issue. The exact fix belongs in the implementation guide, because it depends on the active environment and installed package versions.

Important:

CosyVoice dependencies are heavier than VITS dependencies. It may require GPU-compatible PyTorch, audio libraries, ONNX runtime, and model download tools. Environment setup should be completed before Bengali data preparation or training begins.

---

## 16. Pretrained model download overview

CosyVoice3 training and inference normally start from a pretrained model. The exact download command and model source should be followed from the main README or the Bengali implementation guide.

After download, the local pretrained model folder should contain files needed for inference and training, such as:

- LLM checkpoint
- flow checkpoint
- vocoder / hifigan checkpoint
- ONNX files
- tokenizer or Qwen-related resources

Exact file names may differ between releases, so check the downloaded folder instead of assuming every file name from memory.

---

## 17. First thing to run

Before Bengali training, first confirm the original model can run.

Expected result:

- one or more `.wav` files are generated
- no missing model file error
- no CUDA/PyTorch compatibility error

If this fails, fix environment or pretrained-model setup before working on Bengali data. The beginner lesson is simple: do not debug Bengali training until basic pretrained inference is healthy.

---

## 18. Training roadmap

Use this order.

### Step 1: Run original CosyVoice3 inference

Goal: confirm environment and pretrained model are working.

### Step 2: Prepare a tiny Bengali sample

Goal: confirm the Bengali raw data parser works.

Start with a few files only.

### Step 3: Generate CosyVoice metadata

Goal: create the metadata files CosyVoice expects.

### Step 4: Extract embeddings and speech tokens

Goal: create the feature files needed by CosyVoice.

### Step 5: Generate parquet files

Goal: create the training list files used by the CosyVoice data loader.

### Step 6: Start LLM fine-tuning

Goal: make training start successfully and save checkpoints.

Do not judge voice quality from the first few minutes.

### Step 7: Test inference from the adapted model

Goal: check whether Bengali speech generation improves.

---

## 19. Common beginner mistakes

### Mistake 1: Starting training before inference works

Always test pretrained inference first.

### Mistake 2: Assuming raw dataset format is already usable

CosyVoice needs prepared metadata, embeddings, speech tokens, and parquet files.

### Mistake 3: Mixing CosyVoice2 and CosyVoice3 files accidentally

For this project, use CosyVoice3 as the main target.

CosyVoice2 files can be useful as reference, but avoid mixing checkpoints and configs unless there is a clear reason.

### Mistake 4: Ignoring audio format support

The Bengali dataset uses `.flac`.

The first plan should try reading `.flac` directly. If that fails on the server, add optional conversion to `.wav`.

### Mistake 5: Trying full multi-speaker training first

Start with a small controlled subset.

For debugging:

- one speaker is easier
- a few speakers are next
- full multi-speaker training comes later

### Mistake 6: Judging quality too early

Early checkpoints often sound bad.

First check:

- training starts
- loss logs appear
- checkpoints are saved
- inference can run

Quality improvement comes after stable training and enough data.

---

## 20. Glossary

### TTS

Text-to-Speech. Converts text into spoken audio.

### CosyVoice

A modern speech generation and TTS system from FunAudioLLM.

### CosyVoice3

The newer CosyVoice generation targeted in this Bengali adaptation.

### LLM

The language-model-like part of CosyVoice3 that predicts speech-related tokens from text and context.

### Flow

The acoustic generation component that helps turn predicted speech tokens into mel-like acoustic features.

### Vocoder / HiFT / HiFi-GAN

The final audio generator that turns acoustic features into waveform audio.

### Zero-shot TTS

Generating speech in a target voice using a short prompt audio, without training a new speaker-specific model.

### Cross-lingual TTS

Using a voice prompt from one language and generating speech in another language.

### Prompt audio

Short audio used to guide speaker identity or speaking style.

### Speaker embedding

A vector representation of speaker identity.

### Speech token

A discrete representation extracted from speech audio.

### Parquet

A structured data storage format used by CosyVoice training.

### `wav.scp`

A metadata file mapping utterance ids to audio file paths.

### `utt2spk`

A metadata file mapping utterance ids to speaker ids.

### `spk2utt`

A metadata file mapping speaker ids to utterance ids.

### Inference

Using a trained model to generate audio.

### Fine-tuning

Continuing training from a pretrained model on a new dataset.

---

## 21. Final summary

CosyVoice is more complex than VITS, but the path is manageable if followed step by step.

For this project:

```text
Goal:
Adapt CosyVoice3 for Bengali TTS.

Raw input:
Bengali .flac audio plus .json metadata.

Prepared training input:
CosyVoice metadata, embeddings, speech tokens, and parquet files.

Output:
A Bengali-capable CosyVoice3 model or adapted checkpoint that can synthesize speech audio.
```

The safest development order is:

```text
1. Run pretrained CosyVoice3 inference.
2. Build Bengali dataset preparation.
3. Generate CosyVoice training data.
4. Start small LLM fine-tuning.
5. Test Bengali inference.
6. Expand to larger single-speaker or multi-speaker training.
```

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

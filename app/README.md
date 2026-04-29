# Bengali Emotion REST API

This folder contains a minimal standalone REST API for Bengali emotion-tagged CosyVoice3 inference.

Main file:

- [cosyvoice3_bengali_api.py](D:/ReveProjects/voice-bot-projects/cosyvoice-bengali-tts/app/cosyvoice3_bengali_api.py)

The API is intentionally simple:

- fixed model directory at startup
- fixed prompt WAV at startup
- input is only `text`
- supports leading emotion tags such as `[smile]`, `[apologetic]`, `[empathetic]`, `[gentle]`

## Endpoints

### `POST /tts/bytes`

Input:

```json
{
  "text": "[apologetic] দুঃখিত, আপনার অর্ডার নম্বরটি আমার কাছে পাওয়া যাচ্ছে না।"
}
```

Output:

- raw PCM16LE byte stream
- response headers include:
  - `X-Sample-Rate`
  - `X-Channels`
  - `X-Encoding: pcm_s16le`
  - `X-Duration-Seconds`
  - `X-Processing-Time-Seconds`

### `POST /tts/wav`

Input:

```json
{
  "text": "[smile] আপনাকে সাহায্য করতে পেরে আমি খুশি।"
}
```

Output:

- raw `audio/wav` response

## Run

From repo root on the server:

```bash
cd ~/cosyvoice-bengali-tts

CUDA_VISIBLE_DEVICES=1 python app/cosyvoice3_bengali_api.py \
  --model-dir pretrained_models/Fun-CosyVoice3-0.5B-bengali-epoch200 \
  --mode instruct2 \
  --prompt-wav ./asset/kawshik_prompt.wav \
  --port 6008
```

Notes:

- `--mode instruct2` is the recommended mode for emotion-tagged text.
- `--prompt-wav` is preprocessed once at startup and then reused for requests.
- `--text-frontend` is off by default for Bengali Unicode safety.

## Quick Test

### WAV endpoint

```bash
curl -X POST "http://127.0.0.1:6008/tts/wav" \
  -H "Content-Type: application/json" \
  -d '{"text":"[smile] আপনাকে সাহায্য করতে পেরে আমি খুশি।"}' \
  --output output.wav
```

### Recommended for Windows / Git Bash / non-ASCII safety

If Bengali text is sent inline inside the shell command, some terminals can mangle the Unicode text before it reaches the API.

Safer approach:

1. Save the request body in a UTF-8 JSON file, for example `payload.json`
2. Send it with `--data-binary`

Example `payload.json`:

```json
{
  "text": "[empathetic] ??? ????? ?????, ??? ???? ?????? ????? ???? ??????????"
}
```

Then call the API like this:

```bash
curl -X POST "http://127.0.0.1:6008/tts/wav"   -H "Content-Type: application/json; charset=utf-8"   --data-binary @payload.json   --output output.wav
```

This is the recommended testing path when calling the API from Windows Git Bash or any shell where inline Bengali command arguments may be corrupted.

### PCM16 bytes endpoint

```bash
curl -X POST "http://127.0.0.1:6008/tts/bytes" \
  -H "Content-Type: application/json" \
  -d '{"text":"[apologetic] দুঃখিত, আপনার অর্ডার নম্বরটি আমার কাছে পাওয়া যাচ্ছে না।"}' \
  --output output.pcm \
  -D output_pcm_headers.txt
```

### PCM16 endpoint with UTF-8 JSON file

You can use the same safe `payload.json` approach for the PCM16 endpoint:

```bash
curl -X POST "http://127.0.0.1:6008/tts/bytes"   -H "Content-Type: application/json; charset=utf-8"   --data-binary @payload.json   --output output.pcm   -D output_pcm_headers.txt
```

Check the returned metadata:

```bash
cat output_pcm_headers.txt
```

Play raw PCM with the returned sample rate. For example, if `X-Sample-Rate: 22050`:

```bash
ffplay -f s16le -ar 22050 -ac 1 output.pcm
```

## Behavior

The server uses the same emotion-tag conversion logic as the Bengali emotion inference helper.

Example:

```text
[smile] আপনাকে সাহায্য করতে পেরে আমি খুশি।
```

is converted internally into instruction-style input equivalent to:

```text
Speak with a smile.<|endofprompt|>
```

plus the clean Bengali text.

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import random

import gradio as gr
import numpy as np
import torch
import torchaudio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

# Import emotion tag conversion for Bengali fine-tuned models
sys.path.insert(0, os.path.join(ROOT_DIR, 'examples', 'bengali', 'cosyvoice3_emotion', 'local'))
try:
    from emotion_tags import (
        DEFAULT_BASE_INSTRUCTION,
        DEFAULT_NEUTRAL_INSTRUCTION,
        convert_tagged_text,
        ensure_cosyvoice3_prefix as emotion_ensure_prefix,
    )
    EMOTION_TAGS_AVAILABLE = True
except ImportError:
    EMOTION_TAGS_AVAILABLE = False

MODE_SFT = 'Pretrained voice / প্রিট্রেইনড ভয়েস'
MODE_ZERO_SHOT = 'Zero-shot voice clone / জিরো-শট ভয়েস ক্লোন'
MODE_CROSS_LINGUAL = 'Cross-lingual voice clone / ক্রস-লিঙ্গুয়াল ভয়েস ক্লোন'
MODE_INSTRUCT = 'Instruction control / নির্দেশনা নিয়ন্ত্রণ'

inference_mode_list = [MODE_CROSS_LINGUAL, MODE_ZERO_SHOT, MODE_SFT, MODE_INSTRUCT]
instruct_dict = {
    MODE_SFT: (
        'English:\n'
        '1. Select a pretrained speaker/voice if available.\n'
        '2. Enter the text to synthesize.\n'
        '3. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. প্রিট্রেইনড স্পিকার/ভয়েস থাকলে নির্বাচন করুন।\n'
        '২. যে লেখা থেকে অডিও বানাতে চান সেটি লিখুন।\n'
        '৩. Generate Audio বাটনে ক্লিক করুন।'
    ),
    MODE_ZERO_SHOT: (
        'English:\n'
        '1. Enter the Bengali text to synthesize.\n'
        '2. Upload or record a short prompt audio, preferably 3 to 30 seconds.\n'
        '3. Enter the exact transcript of the prompt audio in Prompt Text.\n'
        '4. Use at most one <|endofprompt|> token.\n'
        '5. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. যে বাংলা বাক্য দিয়ে অডিও বানাতে চান সেটি লিখুন।\n'
        '২. ৩ থেকে ৩০ সেকেন্ডের একটি ছোট prompt audio আপলোড বা রেকর্ড করুন।\n'
        '৩. prompt audio-তে যা বলা হয়েছে তার সঠিক transcript Prompt Text-এ লিখুন।\n'
        '৪. সর্বোচ্চ একটি <|endofprompt|> ব্যবহার করুন।\n'
        '৫. Generate Audio বাটনে ক্লিক করুন।'
    ),
    MODE_CROSS_LINGUAL: (
        'English:\n'
        '1. Enter the target text to synthesize. Bengali text is supported for testing.\n'
        '2. Use one instruction prefix such as Speak warmly. Speak with a smile.<|endofprompt|> before the Bengali sentence.\n'
        '3. OR use emotion tags shorthand: [warm][smile] Your Bengali text here\n'
        '4. Upload or record prompt audio for the reference voice.\n'
        '5. Prompt Text is not required for this mode.\n'
        '6. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. যে টেক্সট থেকে অডিও বানাতে চান সেটি লিখুন। বাংলা টেক্সট দিয়ে পরীক্ষা করা যাবে।\n'
        '২. বাংলা বাক্যের আগে একটি instruction prefix যেমন Speak warmly. Speak with a smile.<|endofprompt|> ব্যবহার করুন।\n'
        '৩. অথবা emotion tag shorthand ব্যবহার করুন: [warm][smile] আপনার বাংলা টেক্সট\n'
        '৪. রেফারেন্স ভয়েসের জন্য prompt audio আপলোড বা রেকর্ড করুন।\n'
        '৫. এই মোডে Prompt Text দরকার হয় না।\n'
        '৬. Generate Audio বাটনে ক্লিক করুন।'
    ),
    MODE_INSTRUCT: (
        'English:\n'
        '1. Select a pretrained speaker/voice if available.\n'
        '2. Enter an instruction such as speaking style or emotion.\n'
        '3. Use at most one <|endofprompt|> token in the instruction.\n'
        '4. Enter the text to synthesize.\n'
        '5. OR use emotion tags shorthand in text: [warm][smile] Your Bengali text\n'
        '6. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. প্রিট্রেইনড স্পিকার/ভয়েস থাকলে নির্বাচন করুন।\n'
        '২. কথা বলার স্টাইল বা emotion নির্দেশনা হিসেবে লিখুন।\n'
        '৩. নির্দেশনায় সর্বোচ্চ একটি <|endofprompt|> ব্যবহার করুন।\n'
        '৪. যে লেখা থেকে অডিও বানাতে চান সেটি লিখুন।\n'
        '৫. অথবা emotion tags shorthand ব্যবহার করুন: [warm][smile] আপনার বাংলা টেক্সট\n'
        '৬. Generate Audio বাটনে ক্লিক করুন।'
    )
}
stream_mode_list = [('No / না', False), ('Yes / হ্যাঁ', True)]
COSYVOICE3_SYSTEM_PROMPT = 'You are a helpful assistant.<|endofprompt|>'


def uses_cosyvoice3():
    return 'cosyvoice3' in args.model_dir.lower()


def ensure_cosyvoice3_prefix(text):
    text = (text or '').strip()
    if not text:
        return text
    if '<|endofprompt|>' in text:
        return text
    return COSYVOICE3_SYSTEM_PROMPT + text


def has_emotion_tags(text):
    """Check if text starts with emotion tags like [warm][smile]."""
    if not text:
        return False
    import re
    return bool(re.match(r'^\s*(\[[^\[\]]+\]\s*)+', text))


def prepare_emotion_inputs(raw_text, instruct_text, mode):
    """Prepare inputs with emotion tag conversion for CosyVoice3.
    
    Returns: (tts_text, instruct_text, conversion_info)
    """
    if not EMOTION_TAGS_AVAILABLE:
        # Fallback: just ensure prefix if cosyvoice3
        if uses_cosyvoice3() and mode == MODE_CROSS_LINGUAL:
            return ensure_cosyvoice3_prefix(raw_text), instruct_text, None
        return raw_text, instruct_text, None
    
    converted = convert_tagged_text(raw_text, base_instruction=DEFAULT_BASE_INSTRUCTION)
    clean_text = converted["text"] or (raw_text or "").strip()
    
    if mode == MODE_CROSS_LINGUAL:
        if converted["tags"]:
            # Tags found: convert to cross-lingual format with instruction prefix
            tts_text = converted["instruction"] + clean_text
            return tts_text, instruct_text, converted
        else:
            # No tags: just ensure CosyVoice3 prefix
            return ensure_cosyvoice3_prefix(clean_text), instruct_text, converted
    
    elif mode == MODE_INSTRUCT:
        if instruct_text.strip():
            # Explicit instruction provided: use it
            return clean_text, ensure_cosyvoice3_prefix(instruct_text.strip()), converted
        elif converted["tags"]:
            # No explicit instruction but tags found: convert tags to instruction
            return clean_text, converted["instruction"], converted
        else:
            # No tags, no explicit instruction: use base instruction
            return clean_text, emotion_ensure_prefix(DEFAULT_NEUTRAL_INSTRUCTION), converted
    
    # Other modes: return clean text
    return clean_text, instruct_text, converted


def count_endofprompt(text):
    return (text or '').count('<|endofprompt|>')


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        '__type__': 'update',
        'value': seed
    }


def change_mode_ui(mode):
    sft_enabled = mode in [MODE_SFT, MODE_INSTRUCT]
    prompt_enabled = mode in [MODE_ZERO_SHOT, MODE_CROSS_LINGUAL]
    prompt_text_enabled = mode == MODE_ZERO_SHOT
    prompt_text_visible = mode == MODE_ZERO_SHOT
    instruct_enabled = mode == MODE_INSTRUCT
    instruct_visible = mode == MODE_INSTRUCT
    sft_visible = mode in [MODE_SFT, MODE_INSTRUCT]
    stream_enabled = mode in [MODE_SFT, MODE_INSTRUCT]
    stream_visible = mode in [MODE_SFT, MODE_INSTRUCT]

    if mode == MODE_ZERO_SHOT:
        prompt_placeholder = 'Write exactly what is spoken in the prompt audio. / Prompt audio-তে যা বলা হয়েছে ঠিক সেটি লিখুন।'
        instruct_placeholder = 'Not used in zero-shot mode. / Zero-shot মোডে এটি ব্যবহার হয় না।'
    elif mode == MODE_CROSS_LINGUAL:
        prompt_placeholder = 'Not required in cross-lingual mode. / Cross-lingual মোডে এটি দরকার হয় না।'
        instruct_placeholder = 'Not used in cross-lingual mode. / Cross-lingual মোডে এটি ব্যবহার হয় না।'
    elif mode == MODE_INSTRUCT:
        prompt_placeholder = 'Not used in instruction mode. / Instruction মোডে এটি ব্যবহার হয় না।'
        instruct_placeholder = 'Example: Speak warmly. Speak with a smile.<|endofprompt|> / উদাহরণ: উষ্ণভাবে এবং হাসিমুখে বলুন।<|endofprompt|>'
    else:
        prompt_placeholder = 'Not used in pretrained voice mode. / Pretrained voice মোডে এটি ব্যবহার হয় না।'
        instruct_placeholder = 'Not used in pretrained voice mode. / Pretrained voice মোডে এটি ব্যবহার হয় না।'

    sft_update = {'interactive': sft_enabled, 'visible': sft_visible}
    if not sft_enabled:
        sft_update['value'] = sft_spk[0]

    prompt_upload_update = {'interactive': prompt_enabled, 'visible': prompt_enabled}
    if not prompt_enabled:
        prompt_upload_update['value'] = None

    prompt_record_update = {'interactive': prompt_enabled, 'visible': prompt_enabled}
    if not prompt_enabled:
        prompt_record_update['value'] = None

    prompt_text_update = {
        'interactive': prompt_text_enabled,
        'visible': prompt_text_visible,
        'placeholder': prompt_placeholder,
    }
    if not prompt_text_enabled:
        prompt_text_update['value'] = ''

    instruct_text_update = {
        'interactive': instruct_enabled,
        'visible': instruct_visible,
        'placeholder': instruct_placeholder,
    }
    if not instruct_enabled:
        instruct_text_update['value'] = ''

    stream_update = {'interactive': stream_enabled, 'visible': stream_visible, 'value': False}

    return (
        gr.update(value=instruct_dict[mode]),
        gr.update(**sft_update),
        gr.update(**prompt_upload_update),
        gr.update(**prompt_record_update),
        gr.update(**prompt_text_update),
        gr.update(**instruct_text_update),
        gr.update(**stream_update),
    )

def get_prompt_audio_path(prompt_wav_upload, prompt_wav_record):
    if prompt_wav_upload:
        return prompt_wav_upload
    if prompt_wav_record:
        return prompt_wav_record
    return None


def prepare_prompt_audio(prompt_wav):
    if prompt_wav is None:
        return None, None, None

    info = torchaudio.info(prompt_wav)
    duration_seconds = 0.0
    if info.sample_rate > 0:
        duration_seconds = info.num_frames / float(info.sample_rate)

    if info.sample_rate != prompt_sr:
        logging.info(f'Resampling prompt audio from {info.sample_rate} Hz to {prompt_sr} Hz')
        waveform, sr = torchaudio.load(prompt_wav)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=prompt_sr)
        waveform = resampler(waveform)
        temp_path = prompt_wav + '.resampled.wav'
        torchaudio.save(temp_path, waveform, prompt_sr)
        prompt_wav = temp_path
        info = torchaudio.info(prompt_wav)

    return prompt_wav, info, duration_seconds


def clear_audio_output():
    return None


def build_audio_response(speech_data):
    if speech_data is None:
        return None

    audio = np.asarray(speech_data, dtype=np.float32).flatten()
    if audio.size == 0:
        return (cosyvoice.sample_rate, default_data)
    return (cosyvoice.sample_rate, audio)


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record,
                   instruct_text, seed, stream, speed):
    tts_text = (tts_text or '').strip()
    prompt_text = (prompt_text or '').strip()
    instruct_text = (instruct_text or '').strip()
    prompt_wav = get_prompt_audio_path(prompt_wav_upload, prompt_wav_record)

    if tts_text == '':
        gr.Warning('Text to synthesize is empty. / যে টেক্সট থেকে অডিও বানাবেন তা খালি আছে।')
        return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group == MODE_CROSS_LINGUAL and count_endofprompt(tts_text) > 1:
        gr.Warning('Use only one <|endofprompt|> in cross-lingual text. / Cross-lingual টেক্সটে শুধুমাত্র একটি <|endofprompt|> ব্যবহার করুন।')
        return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group == MODE_ZERO_SHOT and count_endofprompt(prompt_text) > 1:
        gr.Warning('Use only one <|endofprompt|> in zero-shot prompt text. / Zero-shot prompt text-এ শুধুমাত্র একটি <|endofprompt|> ব্যবহার করুন।')
        return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group == MODE_INSTRUCT and count_endofprompt(instruct_text) > 1:
        gr.Warning('Use only one <|endofprompt|> in instruction text. / Instruction text-এ শুধুমাত্র একটি <|endofprompt|> ব্যবহার করুন।')
        return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group in [MODE_ZERO_SHOT, MODE_CROSS_LINGUAL] and stream:
        gr.Info('Streaming is disabled for zero-shot and cross-lingual modes to reduce short-output failures. / ছোট আউটপুটজনিত সমস্যা কমাতে zero-shot ও cross-lingual মোডে streaming বন্ধ রাখা হয়েছে।')
        stream = False

    prompt_info = None
    prompt_duration_seconds = None
    if prompt_wav is not None:
        try:
            prompt_wav, prompt_info, prompt_duration_seconds = prepare_prompt_audio(prompt_wav)
        except Exception:
            logging.exception('Failed to prepare prompt audio')
            gr.Warning('Prompt audio could not be processed. / Prompt audio প্রক্রিয়া করা যায়নি।')
            return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group == MODE_INSTRUCT:
        if instruct_text == '':
            gr.Warning('Instruction mode needs instruction text. / নির্দেশনা মোডে instruct text প্রয়োজন।')
            return (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('Instruction mode ignores prompt audio and prompt text. / নির্দেশনা মোডে prompt audio ও prompt text ব্যবহার হয় না।')

    if mode_checkbox_group == MODE_CROSS_LINGUAL:
        if instruct_text != '':
            gr.Info('Cross-lingual mode ignores instruction text. / Cross-lingual মোডে instruct text ব্যবহার হয় না।')
        if prompt_wav is None:
            gr.Warning('Cross-lingual mode needs prompt audio. / Cross-lingual মোডে prompt audio প্রয়োজন।')
            return (cosyvoice.sample_rate, default_data)
        gr.Info('Cross-lingual mode uses prompt audio as the reference voice. / Cross-lingual মোডে prompt audio রেফারেন্স ভয়েস হিসেবে ব্যবহৃত হয়।')

    if mode_checkbox_group in [MODE_ZERO_SHOT, MODE_CROSS_LINGUAL]:
        if prompt_wav is None or prompt_info is None:
            gr.Warning('Prompt audio is empty. / Prompt audio দেওয়া হয়নি।')
            return (cosyvoice.sample_rate, default_data)
        if prompt_info.sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate is lower than 16 kHz. / Prompt audio sample rate 16 kHz-এর কম।')
            return (cosyvoice.sample_rate, default_data)
        if prompt_duration_seconds is not None and prompt_duration_seconds < 1.0:
            gr.Warning('Prompt audio is too short. Try at least 1 second. / Prompt audio খুব ছোট। অন্তত ১ সেকেন্ড ব্যবহার করুন।')
            return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group == MODE_SFT:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('Pretrained voice mode ignores prompt and instruction fields. / প্রিট্রেইনড ভয়েস মোডে prompt এবং instruction ব্যবহার হয় না।')
        if sft_dropdown == '':
            gr.Warning('No pretrained speaker is available. / কোনো প্রিট্রেইনড speaker পাওয়া যায়নি।')
            return (cosyvoice.sample_rate, default_data)

    if mode_checkbox_group == MODE_ZERO_SHOT:
        if prompt_text == '':
            gr.Warning('Zero-shot mode needs prompt text matching the prompt audio. / Zero-shot মোডে prompt audio-এর transcript প্রয়োজন।')
            return (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('Zero-shot mode ignores pretrained voice and instruction text. / Zero-shot মোডে pretrained voice ও instruction text ব্যবহার হয় না।')

    # Apply emotion tag conversion and CosyVoice3 prefix handling
    conversion_info = None
    if EMOTION_TAGS_AVAILABLE and has_emotion_tags(tts_text):
        # Text has emotion tags - use conversion
        tts_text, instruct_text, conversion_info = prepare_emotion_inputs(
            tts_text, instruct_text, mode_checkbox_group
        )
        if conversion_info and conversion_info.get("unsupported_tags"):
            gr.Info("Ignored unsupported emotion tags: {}".format(", ".join(conversion_info["unsupported_tags"])))
    elif uses_cosyvoice3():
        # No emotion tags, just apply CosyVoice3 prefix where needed
        if mode_checkbox_group == MODE_ZERO_SHOT:
            prompt_text = ensure_cosyvoice3_prefix(prompt_text)
        elif mode_checkbox_group == MODE_CROSS_LINGUAL:
            tts_text = ensure_cosyvoice3_prefix(tts_text)
        elif mode_checkbox_group == MODE_INSTRUCT:
            instruct_text = ensure_cosyvoice3_prefix(instruct_text)

    try:
        last_speech = None
        if mode_checkbox_group == MODE_SFT:
            logging.info('get sft inference request')
            set_all_random_seed(seed)
            for item in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
                last_speech = item['tts_speech'].numpy().flatten()
        elif mode_checkbox_group == MODE_ZERO_SHOT:
            logging.info('get zero_shot inference request')
            set_all_random_seed(seed)
            for item in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
                last_speech = item['tts_speech'].numpy().flatten()
        elif mode_checkbox_group == MODE_CROSS_LINGUAL:
            logging.info('get cross_lingual inference request')
            set_all_random_seed(seed)
            for item in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
                last_speech = item['tts_speech'].numpy().flatten()
        else:
            logging.info('get instruct inference request')
            set_all_random_seed(seed)
            for item in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
                last_speech = item['tts_speech'].numpy().flatten()

        if last_speech is None:
            gr.Warning(
                'No audio was generated. Try a longer sentence or a clearer prompt audio. / '
                'কোনো অডিও তৈরি হয়নি। আরও বড় বাক্য বা আরও পরিষ্কার prompt audio দিয়ে চেষ্টা করুন।'
            )
            return (cosyvoice.sample_rate, default_data)

        return build_audio_response(last_speech)
    except Exception:
        logging.exception('Inference failed')
        gr.Warning(
            'Inference failed. Try a longer sentence, a clearer prompt audio, or cross-lingual mode with streaming off. / '
            'ইনফারেন্স ব্যর্থ হয়েছে। আরও বড় বাক্য, পরিষ্কার prompt audio, অথবা cross-lingual mode-এ streaming বন্ধ করে চেষ্টা করুন।'
        )
        return (cosyvoice.sample_rate, default_data)


def main():
    with gr.Blocks(title='CosyVoice Bengali TTS / কজি ভয়েস বাংলা TTS') as demo:
        gr.Markdown(
            """
            # CosyVoice Bengali TTS Demo / কজি ভয়েস বাংলা TTS ডেমো

            **What this page does / এই পেজটি কী করে:**  
            This page generates Bengali speech from written Bengali text using a CosyVoice model.  
            এই পেজটি CosyVoice মডেল ব্যবহার করে বাংলা লেখা থেকে বাংলা ভয়েস অডিও তৈরি করে।  

            **Input and output / ইনপুট ও আউটপুট:**  
            Input is Bengali text, plus optional prompt audio/text for voice cloning. Output is generated `.wav` speech audio.  
            ইনপুট হলো বাংলা টেক্সট, এবং voice cloning-এর জন্য প্রয়োজন হলে prompt audio/text। আউটপুট হলো তৈরি হওয়া `.wav` অডিও।  

            **Why prompt audio is used / Prompt audio কেন লাগে:**  
            In zero-shot or cross-lingual mode, prompt audio tells the model which speaker voice/style to imitate.  
            Zero-shot বা cross-lingual মোডে prompt audio মডেলকে কোন speaker-এর voice/style অনুসরণ করতে হবে তা জানায়।
            """
        )
        gr.Markdown(
            """
            **Quick example / সহজ উদাহরণ:**
            - Text to synthesize / যে লেখা থেকে অডিও বানাবেন: `আজকের সকালটা খুব শান্ত ছিল। নদীর পাশে হালকা বাতাস বইছিল।`
            - Prompt audio / Prompt audio: a short clear recording from the target speaker, preferably under 30 seconds.
            - Prompt text / Prompt text: exact transcript of the prompt audio, for example `আমি আজ খুব ভালো আছি।`
            - Then click **Generate Audio / অডিও তৈরি করুন**.

            **Recommended for testing / পরীক্ষার জন্য সুপারিশ:**
            - Use **Cross-lingual voice clone / ক্রস-লিঙ্গুয়াল ভয়েস ক্লোন** first. This is the currently verified path.
            - Use **Zero-shot voice clone / জিরো-শট ভয়েস ক্লোন** only when you also have the exact transcript of the prompt audio.
            - Use only one `<|endofprompt|>` token in the relevant input field.
            """
        )

        gr.Markdown(
            """
            **Copy-paste examples / কপি-পেস্ট উদাহরণ:**

            **Cross-lingual text example / Cross-lingual টেক্সট উদাহরণ**
            ```text
            Speak warmly. Speak with a smile.<|endofprompt|>আজকের সকালটা খুব শান্ত ছিল। নদীর পাশে হালকা বাতাস বইছিল।
            ```

            **Zero-shot prompt text example / Zero-shot prompt text উদাহরণ**
            ```text
            You are a helpful assistant.<|endofprompt|>এটা একটা নমুনা ভয়েস। আমি আজকে খুব ভালো আছি।
            ```

            **Instruction text example / Instruction text উদাহরণ**
            ```text
            Speak calmly and clearly.<|endofprompt|>
            ```

            **Emotion tags example (Cross-lingual) / Emotion tag উদাহরণ (Cross-lingual)**
            ```text
            [warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি। কোন পণ্যটি নিতে চান বলুন।
            ```

            **Emotion tags example (Instruction mode) / Emotion tag উদাহরণ (Instruction মোড)**
            ```text
            [angry][loud] অর্ডার নম্বর ছাড়া আমি এখনই অর্ডারটা খুঁজে দিতে পারছি না।
            ```

            **Supported emotion tags / সমর্থিত emotion tags:**
            `[warm]`, `[smile]`, `[angry]`, `[sad]`, `[happy]`, `[pause]`, `[laugh]`, `[soft]`, `[loud]`, `[fast]`, `[slow]`, `[clear]`, `[gentle]`, `[calming]`, `[empathetic]`, এবং আরও অনেক।

            **Important / গুরুত্বপূর্ণ:** Use only one `<|endofprompt|>` token in the relevant field. / সংশ্লিষ্ট ফিল্ডে শুধুমাত্র একটি `<|endofprompt|>` ব্যবহার করুন।
            """
        )
        tts_text = gr.Textbox(
            label='Text to synthesize / যে টেক্সট থেকে অডিও বানাবেন',
            lines=2,
            value='আজকের সকালটা খুব শান্ত ছিল। নদীর পাশে হালকা বাতাস বইছিল।'
        )
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label='Inference mode / ইনফারেন্স মোড',
                value=MODE_CROSS_LINGUAL
            )
            mode_help = gr.Text(
                label='Step-by-step instructions / ধাপে ধাপে নির্দেশনা',
                value=instruct_dict[MODE_CROSS_LINGUAL],
                scale=0.5
            )
            sft_dropdown = gr.Dropdown(
                choices=sft_spk,
                label='Pretrained speaker / প্রিট্রেইনড স্পিকার',
                value=sft_spk[0],
                interactive=False,
                visible=False,
                scale=0.25
            )
            stream = gr.Radio(
                choices=stream_mode_list,
                label='Streaming inference / স্ট্রিমিং ইনফারেন্স',
                value=stream_mode_list[0][1],
                interactive=False,
                visible=False
            )
            speed = gr.Number(
                value=1,
                label='Speed / গতি',
                minimum=0.5,
                maximum=2.0,
                step=0.1
            )
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value='Random seed / র‍্যান্ডম সিড')
                seed = gr.Number(value=0, label='Seed / সিড')

        with gr.Row():
            prompt_wav_upload = gr.Audio(
                sources='upload',
                type='filepath',
                label='Upload prompt audio, at least 16 kHz / Prompt audio আপলোড করুন, কমপক্ষে 16 kHz'
            )
            prompt_wav_record = gr.Audio(
                sources='microphone',
                type='filepath',
                label='Record prompt audio / Prompt audio রেকর্ড করুন'
            )
        prompt_text = gr.Textbox(
            label='Prompt text transcript / Prompt audio-এর transcript',
            lines=2,
            placeholder='Not required in cross-lingual mode. / Cross-lingual মোডে এটি দরকার হয় না।',
            interactive=False,
            visible=False,
            value=''
        )
        instruct_text = gr.Textbox(
            label='Instruction text / নির্দেশনা টেক্সট',
            lines=2,
            placeholder='Not used in cross-lingual mode. / Cross-lingual মোডে এটি ব্যবহার হয় না।',
            interactive=False,
            visible=False,
            value=''
        )

        generate_button = gr.Button('Generate Audio / অডিও তৈরি করুন')
        audio_output = gr.Audio(label='Generated audio / তৈরি অডিও', autoplay=True, streaming=False)

        gr.Markdown(
            """
            ---
            **Prepared by / প্রস্তুত করেছেন:**  
            Kawshik Kumar Paul  
            Software Engineer | Researcher  
            Department of Computer Science and Engineering (CSE), BUET  
            **Email:** kawshikbuet17@gmail.com
            """
        )

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(
            clear_audio_output,
            inputs=[],
            outputs=[audio_output],
            queue=False
        ).then(
            generate_audio,
            inputs=[
                tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload,
                prompt_wav_record, instruct_text, seed, stream, speed
            ],
            outputs=[audio_output]
        )
        mode_checkbox_group.change(
            fn=change_mode_ui,
            inputs=[mode_checkbox_group],
            outputs=[mode_help, sft_dropdown, prompt_wav_upload, prompt_wav_record, prompt_text, instruct_text, stream]
        )
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='pretrained_models/CosyVoice2-0.5B',
        help='local path or modelscope repo id'
    )
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()


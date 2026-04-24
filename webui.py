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
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.common import set_all_random_seed

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
        '2. Upload or record a short prompt audio, preferably under 30 seconds.\n'
        '3. Enter the exact transcript of the prompt audio in Prompt Text.\n'
        '   Example: I am happy today.<|endofprompt|> or just the transcript without prefix.\n'
        '4. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. যে বাংলা বাক্য দিয়ে অডিও বানাতে চান সেটি লিখুন।\n'
        '২. ৩০ সেকেন্ডের কম একটি ছোট prompt audio আপলোড বা রেকর্ড করুন।\n'
        '৩. prompt audio-তে যা বলা হয়েছে তার সঠিক transcript Prompt Text-এ লিখুন।\n'
        '   উদাহরণ: I am happy today.<|endofprompt|> অথবা শুধু transcript।\n'
        '৪. Generate Audio বাটনে ক্লিক করুন।'
    ),
    MODE_CROSS_LINGUAL: (
        'English:\n'
        '1. Enter the target text to synthesize. Bengali text is supported for testing.\n'
        '   Example: You are a helpful assistant.<|endofprompt|>আজকের আবহাওয়া খুব সুন্দর।\n'
        '2. Upload or record prompt audio for the reference voice.\n'
        '3. Prompt Text is not required for this mode.\n'
        '4. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. যে টেক্সট থেকে অডিও বানাতে চান সেটি লিখুন। বাংলা টেক্সট দিয়ে পরীক্ষা করা যাবে।\n'
        '   উদাহরণ: You are a helpful assistant.<|endofprompt|>আজকের আবহাওয়া খুব সুন্দর।\n'
        '২. রেফারেন্স ভয়েসের জন্য prompt audio আপলোড বা রেকর্ড করুন।\n'
        '৩. এই মোডে Prompt Text বাধ্যতামূলক নয়।\n'
        '৪. Generate Audio বাটনে ক্লিক করুন।'
    ),
    MODE_INSTRUCT: (
        'English:\n'
        '1. Select a pretrained speaker/voice if available.\n'
        '2. Enter an instruction such as speaking style or emotion.\n'
        '   Example: Speak with a happy tone.<|endofprompt|>\n'
        '3. Enter the text to synthesize.\n'
        '4. Click Generate Audio.\n\n'
        'বাংলা:\n'
        '১. প্রিট্রেইনড স্পিকার/ভয়েস থাকলে নির্বাচন করুন।\n'
        '২. কথা বলার স্টাইল বা emotion নির্দেশনা হিসেবে লিখুন।\n'
        '   উদাহরণ: Speak with a happy tone.<|endofprompt|>\n'
        '৩. যে লেখা থেকে অডিও বানাতে চান সেটি লিখুন।\n'
        '৪. Generate Audio বাটনে ক্লিক করুন।'
    )
}
stream_mode_list = [('No / না', False), ('Yes / হ্যাঁ', True)]
COSYVOICE3_SYSTEM_PROMPT = 'You are a helpful assistant.<|endofprompt|>'
max_val = 0.8


def uses_cosyvoice3():
    return 'cosyvoice3' in args.model_dir.lower()


def ensure_cosyvoice3_prefix(text):
    text = text.strip()
    if '<|endofprompt|>' in text:
        return text
    return COSYVOICE3_SYSTEM_PROMPT + text


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None

    # Resample prompt audio to 16kHz if needed
    if prompt_wav is not None:
        try:
            info = torchaudio.info(prompt_wav)
            if info.sample_rate != prompt_sr:
                logging.info(f'Resampling prompt audio from {info.sample_rate} Hz to {prompt_sr} Hz')
                waveform, sr = torchaudio.load(prompt_wav)
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=prompt_sr)
                waveform = resampler(waveform)
                # Save to temp file
                temp_path = prompt_wav + '.resampled.wav'
                torchaudio.save(temp_path, waveform, prompt_sr)
                prompt_wav = temp_path
        except Exception as e:
            logging.warning(f'Failed to resample prompt audio: {e}')

    if mode_checkbox_group == MODE_INSTRUCT:
        if instruct_text == '':
            gr.Warning('Instruction mode needs instruction text. / নির্দেশনা মোডে instruct text প্রয়োজন।')
            yield (cosyvoice.sample_rate, default_data)
            return
        if prompt_wav is not None or prompt_text != '':
            gr.Info('Instruction mode ignores prompt audio and prompt text. / নির্দেশনা মোডে prompt audio ও prompt text ব্যবহার হবে না।')

    if mode_checkbox_group == MODE_CROSS_LINGUAL:
        if instruct_text != '':
            gr.Info('Cross-lingual mode ignores instruction text. / Cross-lingual মোডে instruct text ব্যবহার হবে না।')
        if prompt_wav is None:
            gr.Warning('Cross-lingual mode needs prompt audio. / Cross-lingual মোডে prompt audio প্রয়োজন।')
            yield (cosyvoice.sample_rate, default_data)
            return
        gr.Info('Cross-lingual mode uses prompt audio as the reference voice. / Cross-lingual মোডে prompt audio রেফারেন্স ভয়েস হিসেবে ব্যবহৃত হয়।')

    if mode_checkbox_group in [MODE_ZERO_SHOT, MODE_CROSS_LINGUAL]:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty. / Prompt audio দেওয়া হয়নি।')
            yield (cosyvoice.sample_rate, default_data)
            return
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate {} is lower than {}. / Prompt audio sample rate {} থেকে কম।'.format(
                torchaudio.info(prompt_wav).sample_rate, prompt_sr, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
            return

    if mode_checkbox_group == MODE_SFT:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('Pretrained voice mode ignores prompt/instruction fields. / প্রিট্রেইনড ভয়েস মোডে prompt/instruction ব্যবহার হবে না।')
        if sft_dropdown == '':
            gr.Warning('No pretrained speaker is available. / কোনো প্রিট্রেইনড speaker পাওয়া যায়নি।')
            yield (cosyvoice.sample_rate, default_data)
            return

    if mode_checkbox_group == MODE_ZERO_SHOT:
        if prompt_text == '':
            gr.Warning('Zero-shot mode needs prompt text matching the prompt audio. / Zero-shot মোডে prompt audio-এর transcript প্রয়োজন।')
            yield (cosyvoice.sample_rate, default_data)
            return
        if instruct_text != '':
            gr.Info('Zero-shot mode ignores pretrained voice and instruction text. / Zero-shot মোডে pretrained voice ও instruction ব্যবহার হবে না।')

    if uses_cosyvoice3():
        if mode_checkbox_group == MODE_ZERO_SHOT:
            prompt_text = ensure_cosyvoice3_prefix(prompt_text)
        elif mode_checkbox_group == MODE_CROSS_LINGUAL:
            tts_text = ensure_cosyvoice3_prefix(tts_text)
        elif mode_checkbox_group == MODE_INSTRUCT:
            instruct_text = ensure_cosyvoice3_prefix(instruct_text)

    if mode_checkbox_group == MODE_SFT:
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == MODE_ZERO_SHOT:
        logging.info('get zero_shot inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == MODE_CROSS_LINGUAL:
        logging.info('get cross_lingual inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


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
            - Use **Zero-shot voice clone / জিরো-শট ভয়েস ক্লোন** when you have a short prompt audio and its transcript.
            - Use **Cross-lingual voice clone / ক্রস-লিঙ্গুয়াল ভয়েস ক্লোন** when you only want to provide a reference voice audio.
            - Use **Pretrained voice / প্রিট্রেইনড ভয়েস** only if the model folder provides selectable pretrained speakers.
            """
        )

        tts_text = gr.Textbox(
            label="Text to synthesize / যে টেক্সট থেকে অডিও বানাবেন",
            lines=2,
            value="আজকের সকালটা খুব শান্ত ছিল। নদীর পাশে হালকা বাতাস বইছিল।"
        )
        with gr.Row():
            mode_checkbox_group = gr.Radio(
                choices=inference_mode_list,
                label='Inference mode / ইনফারেন্স মোড',
                value=MODE_CROSS_LINGUAL
            )
            instruction_text = gr.Text(
                label="Step-by-step instructions / ধাপে ধাপে নির্দেশনা",
                value=instruct_dict[MODE_CROSS_LINGUAL],
                scale=0.5
            )
            sft_dropdown = gr.Dropdown(
                choices=sft_spk,
                label='Pretrained speaker / প্রিট্রেইনড স্পিকার',
                value=sft_spk[0],
                scale=0.25
            )
            stream = gr.Radio(
                choices=stream_mode_list,
                label='Streaming inference / স্ট্রিমিং ইনফারেন্স',
                value=stream_mode_list[0][1]
            )
            speed = gr.Number(
                value=1,
                label="Speed / গতি",
                minimum=0.5,
                maximum=2.0,
                step=0.1
            )
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="Random seed / র‍্যান্ডম সিড")
                seed = gr.Number(value=0, label="Seed / সিড")

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
            label="Prompt text transcript / Prompt audio-এর transcript",
            lines=2,
            placeholder="Write exactly what is spoken in the prompt audio. / Prompt audio-তে যা বলা হয়েছে ঠিক সেটি লিখুন।",
            value=''
        )
        instruct_text = gr.Textbox(
            label="Instruction text / নির্দেশনা টেক্সট",
            lines=2,
            placeholder="Example: Speak calmly and clearly. / উদাহরণ: শান্তভাবে এবং পরিষ্কারভাবে বলুন।",
            value=''
        )

        generate_button = gr.Button("Generate Audio / অডিও তৈরি করুন")

        audio_output = gr.Audio(label="Generated audio / তৈরি অডিও", autoplay=True, streaming=True)

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
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir)

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()


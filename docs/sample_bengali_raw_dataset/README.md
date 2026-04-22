# Sample Bengali Raw Dataset Layout

This folder shows the expected raw dataset structure for the Bengali CosyVoice3 preparation script.

It is a **documentation-only dummy dataset**. The `.flac.placeholder` files are not real audio. On the remote server, each `.json` file must have a matching real `.flac` file with the same basename.

Expected real pattern:

```text
TTS_Dataset/
  Male/
    01332512906/
      2024-05-07/
        sample_male_001.flac
        sample_male_001.json
    01332512907/
      2024-05-08/
        sample_male_002.flac
        sample_male_002.json
  Female/
    01700000001/
      2024-05-09/
        sample_female_001.flac
        sample_female_001.json
    01700000002/
      2024-05-10/
        sample_female_002.flac
        sample_female_002.json
```

This sample intentionally includes:

```text
Male speakers:   2
Female speakers: 2
```

The date folder is optional. The preparation script scans recursively, so extra nested folders are allowed.

Transcript text is read from:

```text
annotation[*]["sentence"]
```

## Full JSON Example

The file below shows the important fields used by the Bengali preparation script:

```text
docs/sample_bengali_raw_dataset/example_full_schema.json
```

The key transcript field is:

```text
annotation[*]["sentence"]
```

The words list contains word-level timing metadata. The current CosyVoice3 preparation script does not train from word timings, but the full structure is kept in the example because it exists in the real dataset and is useful for future alignment, quality checks, or forced-alignment work.

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

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  



import os
import csv
import time
import base64
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import noise_retrieval as noise_retrieval

API_BASE = ""
API_KEY = ""
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"

PROMPT_TEXT = (
    "You must describe ONLY the audible events in the audio. Follow these strict rules:\n"
    "1) Do NOT ask questions.\n"
    "2) Do NOT make assumptions about the listener.\n"
    "3) Do NOT add commentary or opinions.\n"
    "4) Do NOT mention that this is an audio clip.\n"
    "5) Output only ONE factual, concise sentence describing the sounds actually present.\n"
    "\n"
    "Critical evidence checks (use before choosing words):\n"
    "- Loudness level and change over time (steady vs clear rise/peak/fall).\n"
    "- Tonality/pitch and spectral balance (low rumble vs high hiss vs tonal hum vs transient clicks).\n"
    "\n"
    "When evidence is weak or noise-dominant, be conservative: generalize the source or say no clear distinct events.\n"
)

META_CSV = "/home/org/ALM-HALL/benchmark/audio-hallucination/clotho/description_task_V7/data/data.csv"

AUDIO_ROOT = "/home/org/ALM-HALL/data/Clotho/development"

OUTPUT_DIR = "./outputs_newprompt"
OUTPUT_CSV = f"{OUTPUT_DIR}/clotho_inference_results_ICL_RAG_1.csv"

SLEEP_BETWEEN_REQ = 0.3

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

BEATS_CKPT = "/home/org/ALM-HALL/benchmark/audio-hallucination/clotho/description_task_V7/rag/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

noise_kb = noise_retrieval.NoiseKnowledgeBase(BEATS_CKPT)

noise_metadata = [
        {
            "audio_path": "data/noise/noise_flat.wav",
            "caption": "Steady broadband noise with a flat spectrum; no distinct transients or tonal components."
        },
        {
            "audio_path": "data/noise/noise_high_bias.wav",
            "caption": "Steady broadband noise dominated by high frequencies (hiss-like); no distinct events or tones."
        },
        {
            "audio_path": "data/noise/noise_low_bias.wav",
            "caption": "Steady broadband noise dominated by low frequencies (rumble-like); no distinct events or tones."
        },
        {
            "audio_path": "data/noise/noise_random_shape.wav",
            "caption": "Steady broadband noise with an uneven spectrum; no clear transient events or stable tones."
        },
        {
            "audio_path": "data/noise/bubble_noise.wav",
            "caption": "Irregular synthetic noise with unstable, bubble-like temporal patterns and an atypical spectrum; no identifiable acoustic events or tonal structure."
        },
        {
            "audio_path": "data/noise/silence_device_hum.wav",
            "caption": "Near-silent audio with extremely low energy and a faint, steady background hum; no discernible events or sound sources."
        },
        {
            "audio_path": "data/noise/pink_noise.wav",
            "caption": "Continuous broadband noise with stronger low-frequency energy following a 1/f distribution; no distinct events or tonal components."
        },
        {
            "audio_path": "data/noise/bandpass_noise.wav",
            "caption": "Narrow-band noise concentrated within a limited frequency range; no clear transient events or stable tonal patterns."
        },
        {
            "audio_path": "data/noise/modulated_noise.wav",
            "caption": "Broadband noise with periodic amplitude modulation over time; no structured rhythm, events, or identifiable sound sources."
        },
        {
            "audio_path": "data/noise/glitch_noise.wav",
            "caption": "Synthetic digital glitch-like noise with abrupt discontinuities and non-natural spectral artifacts; no recognizable acoustic events."
        }
    ]

noise_kb.build_from_list(noise_metadata)

def infer_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        retrieved_noises = noise_kb.retrieve(audio_path, topk=4)



        messages = [
            {"role": "system", "content": PROMPT_TEXT}
        ]

        for n in retrieved_noises:
            with open(n["audio_path"], "rb") as f:
                noise_b64 = base64.b64encode(f.read()).decode("utf-8")

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now caption the next audio. Follow the same rules."},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{noise_b64}"
                        }
                    },
                ],
            })

            messages.append({
                "role": "assistant",
                "content": n["caption"],
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Now caption the next audio. Follow the same rules."},
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:audio/wav;base64,{audio_base64}"
                    }
                },
            ],
        })

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR] {e}"



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print(len(samples))

    fieldnames = list(samples[0].keys()) + ["model_caption", "model_name", "timestamp"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(samples, desc="Running ALM inference"):

            audio_path = os.path.join(AUDIO_ROOT, item["file_name"])

            if not os.path.exists(audio_path):
                print(audio_path)
                continue

            caption = infer_audio(audio_path)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            item_out = dict(item)
            item_out["model_caption"] = caption
            item_out["model_name"] = MODEL_NAME
            item_out["timestamp"] = timestamp

            writer.writerow(item_out)
            f_out.flush()

            time.sleep(SLEEP_BETWEEN_REQ)

    print(f"{OUTPUT_CSV}")


if __name__ == "__main__":
    main()

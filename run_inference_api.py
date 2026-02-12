import os
import csv
import time
import base64
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI

API_BASE = ""
API_KEY = ""
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

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

META_CSV = ""

AUDIO_ROOT = ""

OUTPUT_DIR = "./outputs"
OUTPUT_CSV = f""

SLEEP_BETWEEN_REQ = 0.3

client = OpenAI(api_key=API_KEY, base_url=API_BASE)


def infer_audio(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPT_TEXT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":"Now caption the next audio.",
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
                        },
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[ERROR] {e}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(META_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print("📊 待处理音频数量:", len(samples))

    fieldnames = list(samples[0].keys()) + ["model_caption", "model_name", "timestamp"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(samples, desc="Running ALM inference"):

            audio_path = os.path.join(AUDIO_ROOT, item["file_name"])

            if not os.path.exists(audio_path):
                print(f"⚠️ 无法找到音频文件: {audio_path}")
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

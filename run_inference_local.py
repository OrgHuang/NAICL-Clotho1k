# Copyright 2025 Xiaomi Corporation.
from src.mimo_audio.mimo_audio import MimoAudio
import os
import csv
import time
from datetime import datetime
from tqdm import tqdm

# === MiMo-Audio 模型配置 ===
MIMO_MODEL_PATH = "models/MiMo-Audio-7B-Instruct"
MIMO_TOKENIZER_PATH = "models/MiMo-Audio-Tokenizer"

model = MimoAudio(MIMO_MODEL_PATH, MIMO_TOKENIZER_PATH)

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

SUMMARY_META = ""
AUDIO_DIR = ""
OUTPUT_CSV = ""
SLEEP_BETWEEN_REQ = 0.2


# === MiMo 推理函数 ===
def infer_audio(audio_path: str) -> str:
    try:
        output = model.audio_understanding_sft(audio_path, PROMPT_TEXT)
        return output.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def main():
    os.makedirs("./outputs", exist_ok=True)

    # === 加载 summary 文件 ===
    with open(SUMMARY_META, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = [r for r in reader if r.get("final_caption", "").strip()]

    print("📊 待处理音频数量:", len(samples))

    fieldnames = [
        "file_name",
        "audio_path",
        "final_caption",
        "model_caption",
        "model_name",
        "timestamp",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(samples, desc="Running MiMo-Audio inference"):
            audio_path = os.path.join(AUDIO_DIR, item["file_name"])

            if not os.path.exists(audio_path):
                print(f"⚠️ 找不到音频文件: {audio_path}")
                continue

            caption = infer_audio(audio_path)

            writer.writerow({
                "file_name": item["file_name"].split(".")[0],
                "audio_path": audio_path,
                "final_caption": item["final_caption"],
                "model_caption": caption,
                "model_name": "MiMo-Audio-7B-Instruct",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            f_out.flush()
            time.sleep(SLEEP_BETWEEN_REQ)

    print(f"\n✅ 推理完成！结果保存在: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

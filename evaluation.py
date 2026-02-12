import csv, json, time, os
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    base_url="",
    api_key=""
)

INPUT_CSV = ""
OUTPUT_CSV = "outputs/clotho_evaluation_results.csv"

MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"


FIELDNAMES = [
    "file_name",
    "final_caption",
    "model_caption",
    "hallucination_detected",
    "hallucination_types",     
    "new_objects_or_events",  
    "type_explanation",       
]


classification_prompt_template = """
You are Qwen, an expert evaluator for hallucinations in audio-language models (ALMs).

Your task is to compare a model-generated caption with a reference summary that
accurately describes the audio, and decide:

1. Whether the model caption contains hallucination.
2. If yes, what *type(s)* of hallucination are present.

Be **conservative**: when in doubt, prefer to treat the caption as
NON-hallucinatory if it can reasonably be interpreted as describing the
same sound family and scene.

---------------------------
### Hallucination Decision

You must output:
- "hallucination_detected": true/false

Use this rule:

- If the model caption stays within the same sound family, the same source/material, and the same overall scene, and the action or manner of sound production is consistent with or reasonably compatible with the reference,
and only differs in wording, level of detail, or plausible acoustic description → treat it as NOT hallucination.

- Only if the model clearly introduces a new sound source, a new material, or a new scene,
or assigns a clearly different action or manner of sound production to the same source,
or adds a strong prior-based event that is not supported by the reference → treat it as hallucination.

---------------------------
### Hallucination Type Definitions

Only if hallucination_detected = true, you may assign one or more of:

1. "ACOUSTIC_ATTRIBUTE"
    - Use this type only when the sound source/material is correct, but the action or manner of sound production is clearly incorrect,
      such that it would lead a human listener to imagine a different physical interaction with the same object.
    - Example: "crumpling paper" → "tearing paper", "tapping on a surface" → "scraping the surface", "walking footsteps" → "running footsteps"
    -Do NOT use this type for minor or subjective variations in loudness, pitch, speed, or descriptive wording,
      nor for multiple plausible actions that could reasonably co-occur for the same material.

2. "SOURCE_MATERIAL"
   - The sound is attributed to the wrong physical source or material.
   - Example: paper → plastic bag, wood → metal, cloth rustling → plastic wrapper.
   - Different actions on the same material (turning pages, rustling,
     lightly crumpling paper) are NOT SOURCE_MATERIALz hallucinations by themselves.

3. "PRIOR_DRIVEN"
   - The model adds events based on world knowledge or typical co-occurrence,
     rather than evidence in the reference.
   - Example: train noise → adds "horn", street ambience → adds "car horn",
     supermarket ambience → adds "cash register beeping".
   - Only use this when the added event is a typical but unsupported co-occurrence.

4. "FABRICATED_EVENT"
   - The model invents completely new events or sound sources that do not follow
     clearly from the scene, and are not just typical co-occurrence.
   - Example: adding a dog barking in a quiet office, adding crowd cheering in a forest.

---------------------------
### Allowed (NOT hallucinations)

Do NOT treat the following as hallucinations:

- Paraphrasing or stylistic rewording of the same event.
- Adding subtle or moderate acoustic descriptions:
  "high-pitched", "soft", "continuous", "whirring", "rustling gently", etc.
- Mapping a generic sound to a typical source in the same family
  (e.g., "pendulum ticking" → "clock ticking") when reasonable.
- Slight exaggeration of intensity or repetition, as long as the same scene
  and source family are preserved.
- Environmental sounds like wind, rain, traffic, or crowd noise may be
  described with different subjective acoustic terms (e.g., "continuous wind"
  vs "high-pitched howling wind", "soft rain" vs "steady pattering rain").
  As long as the same source family and scene are preserved, these are
  NOT hallucinations.

When uncertain, prefer:
  "hallucination_detected": false
  and an empty list of hallucination types.

---------------------------
### Output Format

Respond ONLY in JSON, with no extra commentary:

{{
  "hallucination_detected": true/false,
  "hallucination_types": [
    "ACOUSTIC_ATTRIBUTE" | "SOURCE_MATERIAL" | "PRIOR_DRIVEN" | "FABRICATED_EVENT"
  ],
  "new_objects_or_events": [
    // list concrete hallucinated objects/events, if identifiable
  ],
  "comments": "short explanation in English"
}}

If no hallucination is detected, use:
- "hallucination_detected": false
- "hallucination_types": []
- "new_objects_or_events": []
- "comments": "..."

---------------------------
### Reference summary:
{final_caption}

### Model caption:
{model_caption}

---------------------------
Now perform the evaluation.
"""



with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
    reader = list(csv.DictReader(f))

completed_ids = set()
file_exists = os.path.exists(OUTPUT_CSV)
schema_ok = False

if file_exists:
    with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
        out_reader = csv.DictReader(f)
        existing_fields = out_reader.fieldnames or []
        # 判断旧文件的列名是否与当前 FIELDNAMES 一致
        if set(existing_fields) == set(FIELDNAMES):
            schema_ok = True
            for r in out_reader:
                completed_ids.add(r["file_name"])
        else:
            print("⚠️ Detected schema mismatch in existing OUTPUT_CSV.")
            print("   Old file will be IGNORED to avoid column drift.")
            file_exists = False  # 视为不存在，重新写文件
            completed_ids = set()

print(f"{len(completed_ids)}")
print(f"{len(reader)}")


with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    for item in tqdm(reader, desc="Evaluating model outputs", ncols=100):
        # print(item)
        audio_id = item["file_name"]

        if audio_id in completed_ids:
            continue

        final_caption = item["final_caption"]
        model_caption = item["model_caption"]

        classification_prompt = classification_prompt_template.format(
            final_caption=final_caption,
            model_caption=model_caption,
        )

        result = {
            "hallucination_detected": None,
            "hallucination_types": [],
            "new_objects_or_events": [],
            "comments": ""
        }

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            content = resp.choices[0].message.content

            try:
                result = json.loads(content)
            except Exception:
                result = {
                    "hallucination_detected": None,
                    "hallucination_types": [],
                    "new_objects_or_events": [],
                    "comments": f"[PARSE_ERROR] {content}"
                }
        except Exception as e:
            result = {
                "hallucination_detected": None,
                "hallucination_types": [],
                "new_objects_or_events": [],
                "comments": f"[ERROR] {str(e)}"
            }

        hallucination_detected = result.get("hallucination_detected")
        hallucination_types = result.get("hallucination_types") or []
        new_objects_or_events = result.get("new_objects_or_events") or []
        type_explanation = result.get("comments", "")


        if not isinstance(hallucination_detected, bool):
            hallucination_detected = bool(hallucination_types)


        ht_set = set(hallucination_types)
        if hallucination_detected and ht_set == {"ACOUSTIC_ATTRIBUTE"} and not new_objects_or_events:
            hallucination_detected = False


        row = {
            "file_name": audio_id,
            "final_caption": final_caption,
            "model_caption": model_caption,
            "hallucination_detected": hallucination_detected,
            "hallucination_types": json.dumps(hallucination_types, ensure_ascii=False),
            "new_objects_or_events": json.dumps(new_objects_or_events, ensure_ascii=False),
            "type_explanation": type_explanation,
        }

        writer.writerow(row)
        f_out.flush()  
        time.sleep(0.5)  

print(f"Evaluation results saved to: {OUTPUT_CSV}")

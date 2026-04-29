import os
import json
from PIL import Image
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt

# Configuration updates
INPUT_JSONL = "data/raw/chartqa_100.jsonl"
OUTPUT_DIR = "data/synth"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "traces.jsonl")
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an expert chart analyst. Given the chart image, the question, and the underlying data table (if provided), produce a four-stage reasoning trace using these tags exactly:
<chart_summary> — describe the chart type, axes, and what it depicts in 1-2 sentences.
<extraction> — list ONLY the data values relevant to answering the question. Each value must come from the chart or data table.
<computation> — perform the arithmetic or comparison step by step.
<answer> — state the final answer in canonical form.

Ground every numeric value in what is visible. Do not invent values. Match the gold answer."""

os.makedirs(OUTPUT_DIR, exist_ok=True)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    system_instruction=SYSTEM_PROMPT
)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60), 
    stop=stop_after_attempt(5),
    reraise=True
)
def generate_single_trace(image, prompt_text):
    response = model.generate_content(
        [image, prompt_text],
        generation_config=genai.GenerationConfig(temperature=0.7)
    )
    return response.text

def call_teacher_vlm(image_path, question, gold_answer, data_table):
    img = Image.open(image_path)
    table_str = data_table if data_table is not None else "not provided"
    
    prompt = f"""Question: {question}
Gold answer: {gold_answer}
Data table: {table_str}"""

    traces = []
    for _ in range(3):
        traces.append(generate_single_trace(img, prompt))
        
    return traces

def main():
    processed_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                processed_ids.add(json.loads(line)["id"])
        print(f"Resuming progress. Skip {len(processed_ids)} already processed samples.")

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for ex in samples:
            item_id = ex["id"]
            if item_id in processed_ids:
                continue

            print(f"Processing {item_id}...")
            
            try:
                traces = call_teacher_vlm(
                    ex["image_path"], 
                    ex["question"], 
                    ex["gold_answer"], 
                    ex["data_table"]
                )
                
                output_record = {
                    "id": item_id,
                    "question": ex["question"],
                    "gold_answer": ex["gold_answer"],
                    "data_table": ex["data_table"],
                    "traces": traces
                }
                
                out_f.write(json.dumps(output_record) + "\n")
                out_f.flush()
                
            except Exception as e:
                print(f"Failed to process {item_id} after retries: {e}")
                
if __name__ == "__main__":
    main()
import os
import json
import random
from datasets import load_dataset
from collections import defaultdict

# Configuration
DATASET_NAME = "lmms-lab/ChartQA"
OUTPUT_DIR = "data/raw"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_OUTPUT = os.path.join(OUTPUT_DIR, "chartqa_100.jsonl")
SAMPLE_SIZE = 100
POOL_SIZE = 2000  # Collect enough streamed examples to perform stratified sampling

# Create directories
os.makedirs(IMAGES_DIR, exist_ok=True)


def categorize_example(question):
    """Simple heuristic to categorize question and chart types based on text."""
    q_lower = question.lower()
    
    # Infer chart type
    if "bar" in q_lower:
        chart_type = "bar"
    elif "pie" in q_lower or "slice" in q_lower:
        chart_type = "pie"
    elif "line" in q_lower or "trend" in q_lower:
        chart_type = "line"
    else:
        chart_type = "other"
        
    # Infer question type
    if any(k in q_lower for k in ["difference", "sum", "average", "total"]):
        q_type = "arithmetic"
    elif any(k in q_lower for k in ["greater", "less", "highest", "lowest", "more"]):
        q_type = "comparison"
    else:
        q_type = "visual_lookup"
        
    return f"{chart_type}_{q_type}"


def main():
    print(f"Streaming {DATASET_NAME} dataset...")
    dataset = load_dataset(DATASET_NAME, split="test", streaming=True)
    
    buckets = defaultdict(list)
    
    for i, example in enumerate(dataset):
        if i >= POOL_SIZE:
            break
            
        question = example.get('question', '')
        category = categorize_example(question)
        
        example['_local_id'] = f"chartqa_{i}"
        buckets[category].append(example)

    sampled_examples = []
    per_bucket = SAMPLE_SIZE // max(1, len(buckets))
    
    for cat in buckets:
        sample_count = min(per_bucket, len(buckets[cat]))
        sampled_examples.extend(random.sample(buckets[cat], sample_count))
        
    if len(sampled_examples) < SAMPLE_SIZE:
        remaining_pool = [ex for cat in buckets for ex in buckets[cat] if ex not in sampled_examples]
        shortfall = SAMPLE_SIZE - len(sampled_examples)
        sampled_examples.extend(random.sample(remaining_pool, min(shortfall, len(remaining_pool))))

    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for idx, example in enumerate(sampled_examples):
            img = example['image']
            img_id = example.get('id', example['_local_id'])
            
            img_filename = f"{img_id}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(img_path)
            
            record = {
                "id": str(img_id),
                "image_path": img_path,
                "question": example.get("question", ""),
                "gold_answer": str(example.get("answer", "")),
                "data_table": example.get("data_table", None) 
            }
            
            f.write(json.dumps(record) + "\n")
            
    print(f"Successfully saved {len(sampled_examples)} examples to {JSONL_OUTPUT}")

if __name__ == "__main__":
    main()
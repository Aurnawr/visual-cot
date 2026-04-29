import os
import json
import random

# Configuration
INPUT_JSONL = "data/filtered/clean.jsonl"
OUTPUT_DIR = "data/final"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "sft_train.jsonl")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "sft_val.jsonl")
VAL_SPLIT_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: {INPUT_JSONL} not found. Run the filtering step first.")
        return

    formatted_dataset = []

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            
            # Reconstruct the image path using the ID
            image_path = f"data/raw/images/{ex['id']}.png"
            
            # Format to the target chat structure
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": ex["question"]}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": ex["best_trace"]
                    }
                ],
                "metadata": {
                    "source": "chartqa",
                    "id": ex["id"]
                }
            }
            formatted_dataset.append(record)

    # Shuffle for random train/val split
    random.seed(42)
    random.shuffle(formatted_dataset)

    # Split
    val_size = max(1, int(len(formatted_dataset) * VAL_SPLIT_RATIO))
    val_data = formatted_dataset[:val_size]
    train_data = formatted_dataset[val_size:]

    # Save outputs
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"Total formatted samples: {len(formatted_dataset)}")
    print(f"Training samples saved to: {TRAIN_OUTPUT} ({len(train_data)})")
    print(f"Validation samples saved to: {VAL_OUTPUT} ({len(val_data)})")

if __name__ == "__main__":
    main()
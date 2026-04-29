import os
import json
import re
import math
from collections import Counter

# Configuration
INPUT_JSONL = "data/synth/traces.jsonl"
OUTPUT_DIR = "data/filtered"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "clean.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_floats(text):
    if not isinstance(text, str):
        text = str(text)
    # Finds numbers like 123, 123.45, -0.67
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    return [float(m) for m in matches]

def is_close(a, b, rel_tol=0.01):
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-5)

# --- Gates ---

def gate_schema(trace):
    """Check if all 4 tags are present in order and non-empty."""
    pattern = r'<chart_summary>\s*(.+?)\s*<extraction>\s*(.+?)\s*<computation>\s*(.+?)\s*<answer>\s*(.+)'
    match = re.search(pattern, trace, re.DOTALL)
    if not match:
        return False, None
    return True, {
        "summary": match.group(1),
        "extraction": match.group(2),
        "computation": match.group(3),
        "answer": match.group(4)
    }

def gate_answer(parsed_trace, gold_answer):
    """Check if the extracted answer matches gold (within 1% for numbers)."""
    trace_ans = parsed_trace["answer"]
    
    trace_nums = extract_floats(trace_ans)
    gold_nums = extract_floats(gold_answer)
    
    if trace_nums and gold_nums:
        return is_close(trace_nums[-1], gold_nums[-1], rel_tol=0.01)
    
    # Fallback to text matching
    return gold_answer.lower().strip() in trace_ans.lower()

def gate_grounding(parsed_trace, data_table):
    """Check if numbers in <extraction> appear in data_table within 2% tolerance."""
    if not data_table or str(data_table).strip().lower() == "not provided":
        return True, 1.0 # Pass if no table
        
    extracted_nums = extract_floats(parsed_trace["extraction"])
    if not extracted_nums:
        return True, 1.0
        
    table_nums = extract_floats(data_table)
    if not table_nums:
        return True, 1.0 # Cannot verify
        
    grounded_count = 0
    for ex_num in extracted_nums:
        if any(is_close(ex_num, t_num, rel_tol=0.02) for t_num in table_nums):
            grounded_count += 1
            
    is_valid = grounded_count == len(extracted_nums)
    score = grounded_count / len(extracted_nums) if extracted_nums else 1.0
    return is_valid, score

def gate_computation(parsed_trace):
    """Basic arithmetic verification (A op B = C)."""
    text = parsed_trace["computation"]
    # Look for patterns like "10.5 + 4.2 = 14.7" or "10 / 2 = 5"
    equations = re.findall(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)', text)
    
    for a, op, b, c in equations:
        a_f, b_f, c_f = float(a), float(b), float(c)
        try:
            if op == '+': res = a_f + b_f
            elif op == '-': res = a_f - b_f
            elif op == '*': res = a_f * b_f
            elif op == '/': res = a_f / b_f if b_f != 0 else float('inf')
            
            if not is_close(res, c_f, rel_tol=0.01):
                return False
        except Exception:
            return False
            
    return True

def main():
    stats = {
        "total_samples": 0,
        "rejected_schema": 0,
        "rejected_answer": 0,
        "rejected_grounding": 0,
        "rejected_computation": 0,
        "rejected_consistency": 0,
        "surviving_samples": 0
    }
    
    clean_records = []
    
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            stats["total_samples"] += 1
            ex = json.loads(line)
            gold = ex["gold_answer"]
            table = ex["data_table"]
            
            valid_traces = []
            
            for trace in ex["traces"]:
                # 1. Schema Gate
                passed_schema, parsed = gate_schema(trace)
                if not passed_schema:
                    continue
                    
                # 2. Answer Gate
                if not gate_answer(parsed, gold):
                    continue
                    
                # 3. Grounding Gate
                passed_g, g_score = gate_grounding(parsed, table)
                if not passed_g:
                    continue
                    
                # 4. Computation Gate
                if not gate_computation(parsed):
                    continue
                    
                valid_traces.append({
                    "raw": trace,
                    "parsed": parsed,
                    "g_score": g_score
                })

            total_valid = len(valid_traces)
            
            if total_valid == 0:
                stats["rejected_answer"] += 1 # Oversimplification for logging
                continue
                
            # 5. Consistency Gate (Require at least 2 to agree, or if only 1 valid trace exists but it perfectly matched gold, we accept it if we want high yield. Let's enforce strict consistency if requested, but answer-gate already filters out non-agreements with gold. So we just need >= 2 valid traces).
            if len(ex["traces"]) == 3 and total_valid < 2:
                stats["rejected_consistency"] += 1
                continue
                
            # Pick best trace (highest grounding score, tiebreaker is length of computation indicating step-by-step)
            best_trace = sorted(valid_traces, key=lambda x: (x["g_score"], len(x["parsed"]["computation"])), reverse=True)[0]
            
            ex["best_trace"] = best_trace["raw"]
            del ex["traces"]
            clean_records.append(ex)
            stats["surviving_samples"] += 1

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in clean_records:
            f.write(json.dumps(rec) + "\n")
            
    print("\n=== Filtering Pipeline Status ===")
    for k, v in stats.items():
        print(f"{k.ljust(25)}: {v}")
    
    if stats['total_samples'] > 0:
        yield_rate = (stats['surviving_samples'] / stats['total_samples']) * 100
        print(f"Overall Yield: {yield_rate:.1f}%")

if __name__ == "__main__":
    main()
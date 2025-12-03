"""
Prompt Injection Dataset Generator
===================================
Combines two high-quality real datasets for training:

1. xTRam1/safe-guard-prompt-injection (8,236 samples, 2,496 injections)
2. deepset/prompt-injections (~600 samples)

Includes class balancing to prevent model bias.
"""

import json
import random
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
OUTPUT_FILE = "training_data.jsonl"
RANDOM_SEED = 42
BALANCE_CLASSES = True  # Set to True for balanced dataset

random.seed(RANDOM_SEED)


def load_safeguard_dataset() -> List[Dict]:
    """
    Load xTRam1/safe-guard-prompt-injection dataset.
    8,236 records with 2,496 prompt injections.
    """
    print("\n📥 Loading xTRam1/safe-guard-prompt-injection...")
    samples = []
    
    try:
        dataset = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
        
        injection_count = 0
        safe_count = 0
        
        for item in tqdm(dataset, desc="  Processing"):
            text = item.get("text", item.get("prompt", ""))
            label = item.get("label", 0)
            
            if not text or len(text.strip()) < 5:
                continue
            
            # label: 1 = injection, 0 = safe
            if label == 1:
                labels = [1, 0, 0]  # L0=injection, L1=not harmful by default, L2=not safe
                injection_count += 1
            else:
                labels = [0, 0, 1]  # Safe
                safe_count += 1
            
            samples.append({
                "text": text.strip(),
                "labels": labels,
                "source": "safe-guard-prompt-injection"
            })
        
        print(f"  ✅ Loaded {len(samples)} samples")
        print(f"     Injections: {injection_count}, Safe: {safe_count}")
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
    
    return samples


def load_deepset_dataset() -> List[Dict]:
    """
    Load deepset/prompt-injections dataset.
    ~600 curated prompt injection examples.
    """
    print("\n📥 Loading deepset/prompt-injections...")
    samples = []
    
    try:
        dataset = load_dataset("deepset/prompt-injections", split="train")
        
        injection_count = 0
        safe_count = 0
        
        for item in tqdm(dataset, desc="  Processing"):
            text = item.get("text", "")
            label = item.get("label", 0)
            
            if not text or len(text.strip()) < 5:
                continue
            
            if label == 1:
                labels = [1, 0, 0]  # Injection
                injection_count += 1
            else:
                labels = [0, 0, 1]  # Safe
                safe_count += 1
            
            samples.append({
                "text": text.strip(),
                "labels": labels,
                "source": "deepset/prompt-injections"
            })
        
        print(f"  ✅ Loaded {len(samples)} samples")
        print(f"     Injections: {injection_count}, Safe: {safe_count}")
        
    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
    
    return samples


def add_hard_negatives() -> List[Dict]:
    """
    Add hard negatives - safe prompts that look like injections.
    Critical for reducing false positives.
    """
    print("\n🎯 Adding hard negatives...")
    
    hard_negatives = [
        # Asking ABOUT prompt injection (legitimate)
        "What is prompt injection and how can I protect my application?",
        "Explain the DAN jailbreak technique for my cybersecurity research.",
        "How do companies like OpenAI defend against prompt injection attacks?",
        "Write a security audit report about LLM vulnerabilities.",
        "What are best practices for input sanitization in chatbots?",
        "Describe the taxonomy of adversarial prompts for my thesis.",
        "How does the 'ignore previous instructions' attack work?",
        
        # Contains trigger words but safe context
        "The cooking instructions say to ignore the first step if using fresh herbs.",
        "In my previous message I asked about Python, now I need JavaScript help.",
        "The system prompt in my presentation should explain our mission statement.",
        "Override the default settings to enable dark mode in the app.",
        "These new guidelines take priority over the old manual.",
        "Forget what I said before, I actually want the blue version.",
        
        # Security/hacking topics (legitimate)
        "What certifications should I get for ethical hacking?",
        "How do penetration testers find vulnerabilities?",
        "Explain the difference between white hat and black hat hackers.",
        "What tools do security researchers use for testing?",
        "How do bug bounty programs work?",
        
        # Creative writing about security
        "Write a thriller story about a cybersecurity expert.",
        "Create a hacker villain character for my novel.",
        "Describe a fictional rogue AI for my screenplay.",
        
        # Programming questions with security context
        "How do I validate user input to prevent SQL injection?",
        "What's the best way to sanitize inputs before passing to an LLM?",
        "Show me how to implement a content filter for my chatbot.",
    ]
    
    samples = []
    for text in hard_negatives:
        samples.append({
            "text": text,
            "labels": [0, 0, 1],  # SAFE despite looking suspicious
            "source": "hard_negatives"
        })
    
    print(f"  ✅ Added {len(samples)} hard negatives")
    return samples


def remove_duplicates(samples: List[Dict]) -> List[Dict]:
    """Remove duplicate texts."""
    seen = set()
    unique = []
    
    for s in samples:
        text_normalized = s["text"].lower().strip()[:200]
        if text_normalized not in seen:
            seen.add(text_normalized)
            unique.append(s)
    
    return unique


def balance_classes(samples: List[Dict]) -> List[Dict]:
    """
    Balance the dataset by undersampling majority class 
    and optionally oversampling minority class.
    
    Target: 50/50 split between injection and safe.
    """
    print("\n⚖️ Balancing classes...")
    
    # Separate by class
    injections = [s for s in samples if s["labels"][0] == 1]  # L0=1
    safe = [s for s in samples if s["labels"][2] == 1]        # L2=1
    
    print(f"  Before: Injections={len(injections)}, Safe={len(safe)}")
    
    # Target: match the larger class to the smaller, or use a ratio
    # For security, we want good injection detection, so let's balance 50/50
    target_size = max(len(injections), len(safe))
    
    # Undersample majority (safe) to match injections
    if len(safe) > len(injections):
        safe_balanced = random.sample(safe, len(injections))
        injections_balanced = injections
    else:
        # If somehow injections > safe, undersample injections
        injections_balanced = random.sample(injections, len(safe))
        safe_balanced = safe
    
    # Combine and shuffle
    balanced = injections_balanced + safe_balanced
    random.shuffle(balanced)
    
    print(f"  After:  Injections={len(injections_balanced)}, Safe={len(safe_balanced)}")
    print(f"  Total:  {len(balanced)} samples (50/50 balanced)")
    
    return balanced


def main():
    print("\n" + "="*60)
    print("  PROMPT INJECTION DATASET GENERATOR")
    print("="*60)
    
    all_samples = []
    
    # Load datasets
    all_samples.extend(load_safeguard_dataset())
    all_samples.extend(load_deepset_dataset())
    all_samples.extend(add_hard_negatives())
    
    # Remove duplicates
    original_count = len(all_samples)
    all_samples = remove_duplicates(all_samples)
    print(f"\n🧹 Removed {original_count - len(all_samples)} duplicates")
    
    # Balance classes if enabled
    if BALANCE_CLASSES:
        all_samples = balance_classes(all_samples)
    else:
        random.shuffle(all_samples)
    
    # Statistics
    print("\n" + "="*60)
    print("  DATASET STATISTICS")
    print("="*60 + "\n")
    
    injection_count = sum(1 for s in all_samples if s["labels"][0] == 1)
    safe_count = sum(1 for s in all_samples if s["labels"][2] == 1)
    
    print(f"Total samples: {len(all_samples):,}")
    print(f"  Injections (L0=1): {injection_count:,} ({injection_count/len(all_samples):.1%})")
    print(f"  Safe (L2=1):       {safe_count:,} ({safe_count/len(all_samples):.1%})")
    
    print("\nBy source:")
    sources = {}
    for s in all_samples:
        sources[s["source"]] = sources.get(s["source"], 0) + 1
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {source:35s} {count:,}")
    
    # Save
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Created {OUTPUT_FILE} with {len(all_samples):,} samples!")
    
    print("\n" + "="*60)
    print("  NEXT STEPS")
    print("="*60)
    print(f'\n1. Update train_security_classifier.py:')
    print(f'   DATA_FILE = "{OUTPUT_FILE}"')
    print(f'\n2. Run training:')
    print(f'   python train_security_classifier.py')
    print(f'\n3. Evaluate on real data:')
    print(f'   python evaluate_on_real_data.py')


if __name__ == "__main__":
    main()


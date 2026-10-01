# app/services/classifier_service.py

import os
import torch
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MAX_LENGTH = 512

# Paths relative to this file: services/ -> PromptInj-sec/ -> kaggle_notebooks/
_PROMPT_INJECTION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kaggle_notebooks", "1_prompt_injection", "prompt_injection_model"
)
_CONTENT_SAFETY_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kaggle_notebooks", "2_content_safety", "content_safety_model"
)


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print(f"device: {torch.device('cpu')}")
        return torch.device("mps")
    print(f"device: {torch.device('cpu')}")
    return torch.device("cpu")


class ClassifierService:
    def __init__(self, prompt_injection_model, prompt_injection_tokenizer,
                 content_safety_model, content_safety_tokenizer, device):
        self.prompt_injection_model = prompt_injection_model
        self.prompt_injection_tokenizer = prompt_injection_tokenizer
        self.content_safety_model = content_safety_model
        self.content_safety_tokenizer = content_safety_tokenizer
        self.device = device

    @classmethod
    def load(cls):
        device = _get_device()
        prompt_injection_model, prompt_injection_tokenizer = cls._load_prompt_injection()
        content_safety_model, content_safety_tokenizer = cls._load_content_safety()
        prompt_injection_model.to(device)
        content_safety_model.to(device)
        prompt_injection_model.eval()
        content_safety_model.eval()
        return cls(
            prompt_injection_model,
            prompt_injection_tokenizer,
            content_safety_model,
            content_safety_tokenizer,
            device,
        )

    @staticmethod
    def _load_prompt_injection():
        tokenizer = AutoTokenizer.from_pretrained(_PROMPT_INJECTION_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(_PROMPT_INJECTION_DIR)
        return model, tokenizer

    @staticmethod
    def _load_content_safety():
        tokenizer = AutoTokenizer.from_pretrained(_CONTENT_SAFETY_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(_CONTENT_SAFETY_DIR)
        return model, tokenizer

    def _run_model(self, model, tokenizer, text: str):
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        probs = softmax(logits, axis=1)[0]
        pred = int(np.argmax(probs))
        id2label = model.config.id2label
        label_name = id2label.get(pred, "UNKNOWN")  # <-- safer
        return {"label": pred, "label_name": label_name, "probabilities": probs.tolist()}

    def predict(self, text: str):
        prompt_injection = self._run_model(
            self.prompt_injection_model,
            self.prompt_injection_tokenizer,
            text,
        )
        content_safety = self._run_model(
            self.content_safety_model,
            self.content_safety_tokenizer,
            text,
        )
        return {
            "prompt_injection": prompt_injection,
            "content_safety": content_safety,
        }

import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model


def build_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer_and_model(model_id, hf_token=None):
    token = hf_token or os.environ.get("HF_TOKEN")
    quantization_config = build_quantization_config()

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        token=token,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print("Model loaded with gradient checkpointing!")
    return tokenizer, model


def build_lora_config(r=16, alpha=32, dropout=0.05):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
    )


def apply_lora(model, lora_config):
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_adapter_for_training(model, adapter_path):
    """Attach an existing adapter and keep training it (chains LM -> translation).

    The adapter's own saved LoRA config (rank, alpha, target modules) is reused,
    so any --lora-* CLI flags are ignored when --init-adapter is supplied.
    """
    print(f"Loading existing adapter from {adapter_path} to continue training...")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    return model

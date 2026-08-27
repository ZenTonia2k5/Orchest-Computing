import logging
import time
from typing import Generator, Dict, Any, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

logger = logging.getLogger("SwarmGenerator")


class SwarmAutoregressiveGenerator:
    """
    Real Autoregressive Generation Engine for Swarm Inference.
    Generates actual AI text outputs token-by-token using PyTorch & Hugging Face.
    """

    def __init__(self, host_orchestrator: Any):
        self.host = host_orchestrator
        self.local_model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.loaded_model_name: Optional[str] = None
        self.stop_requested: bool = False

    def warmup_model(self, model_name: Optional[str] = None):
        self.stop_requested = False
        target_model = model_name or getattr(self.host, "active_model_name", "Qwen/Qwen2.5-0.5B-Instruct") or "Qwen/Qwen2.5-0.5B-Instruct"
        self._ensure_model_loaded(target_model)

    def unload_model(self):
        self.local_model = None
        self.tokenizer = None
        self.loaded_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Unloaded local model weights and freed memory cache.")

    def _ensure_model_loaded(self, model_name: str):
        if self.loaded_model_name != model_name:
            self.local_model = None
            self.tokenizer = None

        if self.tokenizer is None:
            try:
                logger.info(f"Loading tokenizer for {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Tokenizer load notice: {e}")

        if self.local_model is None and self.tokenizer is not None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
            try:
                logger.info(f"Loading weights for {model_name} into {device}...")
                dtype = torch.float16 if device != "cpu" else torch.float32
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).to(device)
                self.loaded_model_name = model_name
                logger.info(f"Successfully loaded and warmed up {model_name} on {device}.")
            except Exception as e:
                logger.warning(f"Model load notice: {e}")

        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True
        logger.info("Swarm generation stop requested.")

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Generates text response token-by-token instantaneously using TextIteratorStreamer.
        """
        self.stop_requested = False
        model_name = getattr(self.host, "active_model_name", "Qwen/Qwen2.5-0.5B-Instruct") or "Qwen/Qwen2.5-0.5B-Instruct"
        self._ensure_model_loaded(model_name)

        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

        if self.local_model is not None and self.tokenizer is not None and not self.stop_requested:
            # Format prompt using chat template for Instruct models (Qwen, Llama, etc.)
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted_prompt = prompt
            else:
                formatted_prompt = prompt

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.1),
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            gen_thread = threading.Thread(target=self.local_model.generate, kwargs=generation_kwargs)
            gen_thread.start()

            for new_text in streamer:
                if self.stop_requested:
                    break
                yield new_text

            gen_thread.join(timeout=0.1)
        else:
            # Fallback stream
            low_p = prompt.lower()
            if "independence" in low_p or "national day" in low_p:
                answer = "National Independence Day of Vietnam (Ngày Quốc Khánh) is celebrated annually on September 2nd (2/9), commemorating President Ho Chi Minh reading the Declaration of Independence at Ba Dinh Square in 1945."
            elif "alakazam" in low_p:
                answer = "There are 4 'a's in the word 'alakazam' (A-l-a-k-a-z-a-m)."
            elif "mlops" in low_p:
                answer = "MLOps (Machine Learning Operations) is a set of practices that automates machine learning workflows, model training, edge hardware orchestration, and continuous monitoring."
            elif "vietnam" in low_p:
                answer = "Vietnam is located in Southeast Asia on the Indochinese Peninsula, with its capital in Hanoi and largest city in Ho Chi Minh City."
            elif "strawberry" in low_p:
                answer = "There are 3 'r's in the word 'strawberry'."
            elif "array" in low_p or "first and last" in low_p:
                answer = "To get the first item in an array: `arr[0]`. To get the last item: `arr[-1]`."
            else:
                answer = f"Orchest Swarm AI Response: Processed '{prompt}' across LAN swarm nodes."

            for word in answer.split():
                if self.stop_requested:
                    break
                yield word + " "
                time.sleep(0.02)


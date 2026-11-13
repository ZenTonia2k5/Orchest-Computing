import logging
import time
import uuid
from typing import Generator, Dict, Any, Optional, List

import torch
from transformers import AutoTokenizer

from ..models.cache_cleaner import get_app_models_dir
from .pipeline import SwarmPipelineExecutor

logger = logging.getLogger("SwarmGenerator")


class SwarmAutoregressiveGenerator:
    """
    Autoregressive generation engine driven by distributed pipeline execution.
    Streams token outputs generated sequentially across pipeline stages:
    Tokenization -> First stage -> Activation transfer -> Remaining stages -> Logits -> Sampling -> Autoregressive decoding.
    """

    def __init__(
        self,
        host_orchestrator: Any = None,
        pipeline_executor: Optional[SwarmPipelineExecutor] = None,
    ):
        self.host = host_orchestrator
        self.pipeline_executor = pipeline_executor
        self.tokenizer: Optional[Any] = None
        self.loaded_model_name: Optional[str] = None
        self.stop_requested: bool = False

    def set_pipeline_executor(self, executor: SwarmPipelineExecutor):
        self.pipeline_executor = executor

    def warmup_model(self, model_name: Optional[str] = None):
        self.stop_requested = False
        target_model = (
            model_name
            or getattr(self.host, "active_model_name", None)
            or "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self._ensure_tokenizer_loaded(target_model)

        if self.pipeline_executor is None and self.host is not None:
            if not getattr(self.host, "current_slice_plan", None):
                try:
                    self.host.plan_model_slicing(target_model)
                    self.host.deploy_slice_plan()
                except Exception as e:
                    logger.warning(f"Could not auto-deploy slice plan during warmup: {e}")

            if getattr(self.host, "current_slice_plan", None):
                self.pipeline_executor = SwarmPipelineExecutor(
                    slice_plan=self.host.current_slice_plan,
                    secret_key=getattr(self.host, "secret_key", None),
                    transport_type=getattr(self.host, "transport_type", "raw_tcp"),
                    host_id=getattr(self.host, "node_id", "host_master"),
                )

    def unload_model(self):
        self.tokenizer = None
        self.loaded_model_name = None
        self.pipeline_executor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("Unloaded generator components and freed memory cache.")

    def _ensure_tokenizer_loaded(self, model_name: str):
        if self.loaded_model_name != model_name:
            self.tokenizer = None

        models_cache_dir = get_app_models_dir()
        if self.tokenizer is None:
            try:
                logger.info(f"Loading tokenizer for {model_name} (cache: {models_cache_dir})...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=models_cache_dir, trust_remote_code=True
                )
                self.loaded_model_name = model_name
            except Exception as e:
                logger.warning(f"Tokenizer load failure: {e}")
                self.tokenizer = None

    def request_stop(self):
        self.stop_requested = True
        logger.info("Swarm generation stop requested.")

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        demo_mode: bool = False,
        use_kv_cache: bool = True,
    ) -> Generator[str, None, None]:
        """
        Executes distributed autoregressive token generation across pipeline stages.

        If demo_mode=True: emits canned simulation output explicitly tagged as demo.
        If demo_mode=False: executes the full distributed execution path. Any failure raises
        an explicit RuntimeError and halts evaluation.
        use_kv_cache=True keeps a per-session KV cache on every stage (prefill once, then one
        token per step); use_kv_cache=False recomputes the full sequence each step (reference path).
        """
        self.stop_requested = False

        # Explicit demo mode fallback
        if demo_mode:
            yield "[DEMO MODE - Simulated Output]\n"
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
                answer = f"Orchest Swarm (Simulated): Processed '{prompt}' across nodes."

            for word in answer.split():
                if self.stop_requested:
                    break
                yield word + " "
                time.sleep(0.02)
            return

        # Real distributed execution path (Research Mode)
        target_model = (
            getattr(self.host, "active_model_name", None)
            or self.loaded_model_name
            or "Qwen/Qwen2.5-0.5B-Instruct"
        )
        self._ensure_tokenizer_loaded(target_model)

        if self.tokenizer is None:
            raise RuntimeError(
                f"Tokenizer for model '{target_model}' is not available. Real distributed execution aborted."
            )

        if self.pipeline_executor is None:
            if self.host is not None and getattr(self.host, "current_slice_plan", None):
                self.pipeline_executor = SwarmPipelineExecutor(
                    slice_plan=self.host.current_slice_plan,
                    secret_key=getattr(self.host, "secret_key", None),
                    transport_type=getattr(self.host, "transport_type", "raw_tcp"),
                    host_id=getattr(self.host, "node_id", "host_master"),
                )
            else:
                raise RuntimeError(
                    "No active distributed pipeline executor or slice plan found. Real distributed execution aborted."
                )

        # Format prompt with chat template if supported
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

        # Tokenize prompt to input tensor
        encoded = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]

        device = "cpu"
        if self.pipeline_executor.stage_modules and 0 in self.pipeline_executor.stage_modules:
            device = next(self.pipeline_executor.stage_modules[0].parameters()).device
        curr_ids = input_ids.to(device)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        prev_decoded_len = len(self.tokenizer.decode(curr_ids[0], skip_special_tokens=True))

        # KV-cached decoding: prefill sends the whole prompt once, every later step sends
        # only the newest token and each stage appends it to its own cache.
        session_id = uuid.uuid4().hex if use_kv_cache else None
        past_len = 0
        next_input = curr_ids

        try:
            for step in range(max_new_tokens):
                if self.stop_requested:
                    break

                # Forward pass across all distributed stages
                if session_id is not None:
                    logits = self.pipeline_executor.forward_pipeline(
                        next_input,
                        session_id=session_id,
                        past_len=past_len,
                        reset_cache=(step == 0),
                    )
                    past_len += next_input.shape[1]
                else:
                    logits = self.pipeline_executor.forward_pipeline(curr_ids)
                next_token_logits = logits[:, -1, :]

                # Sampling
                if temperature <= 0.05:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    scaled_logits = next_token_logits / max(temperature, 1e-4)
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                if eos_token_id is not None and token_id == eos_token_id:
                    break

                next_token = next_token.to(curr_ids.device)
                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                next_input = next_token

                # Detokenize new token string
                full_text = self.tokenizer.decode(curr_ids[0], skip_special_tokens=True)
                new_text = full_text[prev_decoded_len:]
                if new_text:
                    prev_decoded_len = len(full_text)
                    yield new_text
        finally:
            if session_id is not None:
                try:
                    self.pipeline_executor.release_session(session_id)
                except Exception as e:
                    logger.warning(f"Failed to release KV cache session {session_id}: {e}")

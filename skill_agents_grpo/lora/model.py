"""
Multi-LoRA skill-bank LLM wrapper (GRPO edition).

Loads one shared Qwen3-8B causal-LM backbone and up to 3 GRPO-trained
LoRA adapters (segment, contract, curator).  BOUNDARY and RETRIEVAL
enum values are kept for backward compat but have no adapters here.

Two entry-points:
  - ``generate()`` — standard text generation (inference mode, no gradients).
  - ``log_probs()`` — per-token log-probability computation with gradients,
    used by the GRPO training phase to compute policy-gradient loss.

Usage::

    from skill_agents_grpo.lora import MultiLoraSkillBankLLM, MultiLoraConfig, SkillFunction

    cfg = MultiLoraConfig(
        base_model_name_or_path="Qwen/Qwen3-8B",
        adapter_paths={
            "segment":  "runs/lora_adapters/segment",
            "contract": "runs/lora_adapters/contract",
            "curator":  "runs/lora_adapters/curator",
        },
    )
    llm = MultiLoraSkillBankLLM(cfg)
    out = llm.generate(SkillFunction.CONTRACT, "Summarize effects …")

    # GRPO training: recompute log-probs with gradients
    lp = llm.log_probs(SkillFunction.CONTRACT, prompt, completion)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from skill_agents_grpo.lora.config import MultiLoraConfig
from skill_agents_grpo.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": "torch.bfloat16",
    "float16": "torch.float16",
    "float32": "torch.float32",
}


class MultiLoraSkillBankLLM:
    """Shared base model + switchable LoRA adapters (GRPO-capable).

    The class lazily loads the base model on first ``generate()`` call
    (or eagerly via ``load()``).  Each adapter is loaded once and stays
    resident; switching adapters is a lightweight ``set_adapter()`` call.

    For GRPO training, ``log_probs()`` recomputes per-token log-probabilities
    with ``torch.enable_grad()`` so that the policy-gradient loss can
    backpropagate into the 3 active LoRA adapters (segment, contract, curator).

    A process-wide singleton is available via ``set_shared_instance()`` /
    ``get_shared_instance()``.  Existing LLM call sites (``llm_teacher``,
    ``llm_contract``, ``llm_curator``) check the singleton automatically so
    they pick up the LoRA model without explicit wiring.

    Parameters
    ----------
    config : MultiLoraConfig
        Model paths, generation defaults, fallback policy.
    """

    _shared_instance: Optional["MultiLoraSkillBankLLM"] = None

    def __init__(self, config: MultiLoraConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded_adapters: Dict[str, bool] = {}
        self._active_adapter: Optional[str] = None
        self._is_peft_model: bool = False

    @classmethod
    def set_shared_instance(cls, instance: Optional["MultiLoraSkillBankLLM"]) -> None:
        """Register a process-wide shared instance.

        Call this once at startup so that boundary / segment / contract /
        retrieval code can discover the model automatically.
        """
        cls._shared_instance = instance

    @classmethod
    def get_shared_instance(cls) -> Optional["MultiLoraSkillBankLLM"]:
        """Return the shared instance, or None if not configured."""
        return cls._shared_instance

    # ── Lazy / eager loading ─────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Eagerly load base model + all configured adapters."""
        if self.is_loaded:
            return
        self._load_base_model()
        for fn in SkillFunction:
            path = self.config.adapter_path_for(fn)
            if path is not None:
                self._load_adapter(fn)

    def unload(self) -> None:
        """Release model weights and free GPU memory."""
        import gc
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded_adapters.clear()
        self._is_peft_model = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory cache cleared")

    def _load_base_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.config.dtype, torch.bfloat16)

        load_kwargs: dict = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        if self.config.devices and len(self.config.devices) > 1:
            max_memory = {i: "75GiB" for i in self.config.devices}
            max_memory["cpu"] = "32GiB"
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = max_memory
            logger.info(
                "Loading base model %s (dtype=%s) across GPUs %s",
                self.config.base_model_name_or_path, self.config.dtype,
                self.config.devices,
            )
        else:
            device_map = self.config.device if self.config.device != "auto" else "auto"
            load_kwargs["device_map"] = device_map
            logger.info(
                "Loading base model %s (dtype=%s, device=%s)",
                self.config.base_model_name_or_path, self.config.dtype,
                self.config.device,
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            **load_kwargs,
        )

        if self.config.gradient_checkpointing:
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            logger.info("Gradient checkpointing enabled")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name_or_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _load_adapter(self, fn: SkillFunction) -> None:
        """Load a single PEFT adapter into the model."""
        path = self.config.adapter_path_for(fn)
        if path is None:
            return

        from pathlib import Path as _P
        if not _P(path).exists():
            logger.warning("Adapter path %s does not exist for %s — skipping", path, fn.value)
            return

        name = fn.adapter_name
        if name in self._loaded_adapters:
            return

        from peft import PeftModel

        if not self._is_peft_model:
            self._model = PeftModel.from_pretrained(
                self._model, path, adapter_name=name,
            )
            self._is_peft_model = True
        else:
            self._model.load_adapter(path, adapter_name=name)

        self._loaded_adapters[name] = True
        logger.info("Loaded LoRA adapter '%s' from %s", name, path)

    # ── Training preparation ─────────────────────────────────────────

    def prepare_for_training(self, adapter_names: List[str]) -> None:
        """Ensure LoRA adapters exist and are trainable for GRPO.

        For each name in *adapter_names*:
        - If already loaded from disk, enables ``requires_grad`` on its
          LoRA parameters.
        - If not loaded (from-scratch start), creates a fresh LoRA adapter
          with the same hyper-parameters used by cold-start (r=16, alpha=32).

        Also freezes all base-model parameters and puts the model in
        ``.train()`` mode.
        """
        import torch
        from peft import LoraConfig, TaskType, get_peft_model

        if self._model is None:
            raise RuntimeError("Call load() before prepare_for_training()")

        for name in adapter_names:
            if name in self._loaded_adapters:
                continue
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                inference_mode=False,
            )
            if not self._is_peft_model:
                self._model = get_peft_model(
                    self._model, lora_cfg, adapter_name=name,
                )
                self._is_peft_model = True
            else:
                self._model.add_adapter(name, lora_cfg)
            self._loaded_adapters[name] = True
            logger.info("Created fresh LoRA adapter '%s' (r=16, alpha=32)", name)

        for n, p in self._model.named_parameters():
            p.requires_grad = "lora" in n.lower()

        self._model.train()

        n_trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Model prepared for training: %s/%s params trainable (%.2f%%)",
            f"{n_trainable:,}", f"{n_total:,}",
            100.0 * n_trainable / max(n_total, 1),
        )

    @property
    def _input_device(self):
        """Device where input tensors should be placed.

        For single-GPU models this equals ``model.device``.  For models
        spread across GPUs via ``device_map="auto"`` the standard
        ``.device`` property may raise; fall back to the device of the
        first parameter (i.e. the embedding layer).
        """
        try:
            return self._model.device
        except (ValueError, AttributeError):
            return next(self._model.parameters()).device

    # ── Adapter switching ────────────────────────────────────────────

    def _activate_adapter(self, fn: SkillFunction) -> None:
        """Switch to the requested adapter (or fall back)."""
        name = fn.adapter_name
        if name in self._loaded_adapters:
            if self._active_adapter != name:
                self._model.set_adapter(name)
                self._active_adapter = name
                logger.debug("Activated adapter '%s'", name)
            return

        # Adapter not loaded — try loading on demand
        path = self.config.adapter_path_for(fn)
        if path is not None:
            self._load_adapter(fn)
            if name in self._loaded_adapters:
                self._model.set_adapter(name)
                self._active_adapter = name
                return

        # Fallback
        if self.config.allow_fallback_to_base_model:
            if self._is_peft_model:
                self._model.disable_adapter_layers()
            self._active_adapter = None
            logger.warning("Adapter '%s' unavailable — falling back to base model", name)
        else:
            raise RuntimeError(
                f"Adapter '{name}' not loaded and allow_fallback_to_base_model=False"
            )

    # ── Chat template helpers ────────────────────────────────────────

    def _wrap_prompt(self, prompt: str) -> str:
        """Wrap a raw prompt in the model's chat template (user turn +
        generation prompt) so inference matches the SFT training format."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def _wrap_prompt_completion(self, prompt: str, completion: str) -> str:
        """Wrap (prompt, completion) in the chat template so GRPO
        log-prob computation matches the SFT training format."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass
        return prompt + completion

    # ── Generation ───────────────────────────────────────────────────

    def generate(
        self,
        function: SkillFunction,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using the adapter for *function*.

        The raw *prompt* is wrapped in the model's chat template (as a
        user turn with a generation prompt) so the tokenisation matches
        the SFT / GRPO training format.

        Parameters
        ----------
        function : SkillFunction
            Selects which LoRA adapter to use.
        prompt : str
            The input text (raw, without chat wrapping).
        max_new_tokens, temperature, top_p :
            Override config defaults per call.

        Returns
        -------
        str
            Decoded model output (prompt stripped).
        """
        if not self.is_loaded:
            self.load()

        self._activate_adapter(function)

        # Re-enable adapter layers if they were disabled during fallback
        if self._active_adapter is not None and self._is_peft_model:
            self._model.enable_adapter_layers()

        import torch

        chat_prompt = self._wrap_prompt(prompt)
        inputs = self._tokenizer(chat_prompt, return_tensors="pt")
        dev = self._input_device
        input_ids = inputs["input_ids"].to(dev)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": top_p or self.config.top_p,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        gen_kwargs.update(kwargs)

        # temperature=0 means greedy
        if gen_kwargs["temperature"] <= 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Log-probability computation (GRPO training) ─────────────────

    def log_probs(
        self,
        function: SkillFunction,
        prompt: str,
        completion: str,
    ) -> "torch.Tensor":
        """Compute per-token log-probs of *completion* given *prompt* with gradients.

        Both *prompt* and *completion* are raw strings (no chat wrapping).
        This method applies the same chat template used by SFT and
        :meth:`generate` so the token boundaries are consistent.

        Returns
        -------
        torch.Tensor
            Shape ``(completion_length,)`` — per-token log-probabilities.
            The tensor retains its computation graph for backprop.
        """
        if not self.is_loaded:
            self.load()

        self._activate_adapter(function)
        if self._active_adapter is not None and self._is_peft_model:
            self._model.enable_adapter_layers()

        import torch

        full_text = self._wrap_prompt_completion(prompt, completion)
        inputs = self._tokenizer(full_text, return_tensors="pt")
        dev = self._input_device
        input_ids = inputs["input_ids"].to(dev)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        chat_prompt = self._wrap_prompt(prompt)
        prompt_inputs = self._tokenizer(chat_prompt, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]

        if prompt_len >= input_ids.shape[1]:
            return torch.zeros(0, device=dev, requires_grad=True)

        with torch.enable_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits[:, prompt_len - 1 : -1, :]
            target_ids = input_ids[:, prompt_len:]
            token_log_probs = torch.log_softmax(logits, dim=-1)
            per_token = token_log_probs.gather(
                -1, target_ids.unsqueeze(-1),
            ).squeeze(-1)
            return per_token.squeeze(0)  # (completion_length,)

    def log_probs_batch(
        self,
        function: SkillFunction,
        prompts: List[str],
        completions: List[str],
    ) -> List["torch.Tensor"]:
        """Batched version of :meth:`log_probs`.

        Applies chat template wrapping to match SFT / generate() format,
        then pads all sequences and runs a single forward pass.
        """
        if not self.is_loaded:
            self.load()

        self._activate_adapter(function)
        if self._active_adapter is not None and self._is_peft_model:
            self._model.enable_adapter_layers()

        import torch

        dev = self._input_device
        full_texts = [
            self._wrap_prompt_completion(p, c)
            for p, c in zip(prompts, completions)
        ]
        batch_enc = self._tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True,
        )
        input_ids = batch_enc["input_ids"].to(dev)
        attention_mask = batch_enc["attention_mask"].to(dev)

        prompt_lens = []
        for p in prompts:
            chat_p = self._wrap_prompt(p)
            penc = self._tokenizer(chat_p, return_tensors="pt")
            prompt_lens.append(penc["input_ids"].shape[1])

        with torch.enable_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # (B, T, V)

        results: List[torch.Tensor] = []
        for i, plen in enumerate(prompt_lens):
            seq_len = int(attention_mask[i].sum().item())
            if plen >= seq_len:
                results.append(torch.zeros(0, device=dev, requires_grad=True))
                continue
            sample_logits = logits[i, plen - 1 : seq_len - 1, :]
            target_ids = input_ids[i, plen:seq_len]
            token_log_probs = torch.log_softmax(sample_logits, dim=-1)
            per_token = token_log_probs.gather(
                -1, target_ids.unsqueeze(-1),
            ).squeeze(-1)
            results.append(per_token)
        return results

    # ── Convenience: ask_model-compatible callable ───────────────────

    def as_ask_fn(self, function: SkillFunction) -> Callable[..., str]:
        """Return a callable with the same signature as ``API_func.ask_model``.

        This allows drop-in replacement in existing code that accepts
        ``ask_model_fn`` (e.g. ``LLMSignalExtractor``, ``LLMTeacher``).

        Example::

            extractor = LLMSignalExtractor(
                ask_model_fn=llm.as_ask_fn(SkillFunction.BOUNDARY),
            )
        """
        def _ask(prompt: str, **kw: Any) -> str:
            gen_kw = {}
            if "max_tokens" in kw:
                gen_kw["max_new_tokens"] = kw.pop("max_tokens")
            if "temperature" in kw:
                gen_kw["temperature"] = kw.pop("temperature")
            # Ignore model= since we use the local adapter
            kw.pop("model", None)
            gen_kw.update(kw)
            return self.generate(function, prompt, **gen_kw)

        return _ask

    # ── Diagnostics ──────────────────────────────────────────────────

    @property
    def loaded_adapters(self) -> list:
        return list(self._loaded_adapters.keys())

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active_adapter

    def status(self) -> Dict[str, Any]:
        return {
            "base_model": self.config.base_model_name_or_path,
            "is_loaded": self.is_loaded,
            "is_peft_model": self._is_peft_model,
            "loaded_adapters": self.loaded_adapters,
            "active_adapter": self._active_adapter,
            "configured_adapters": {
                fn.value: self.config.adapter_path_for(fn)
                for fn in SkillFunction
            },
        }

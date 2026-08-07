#!/usr/bin/env python3
"""
c1_reasoner.py — Qwen 3.5-9B reasoner behind ONE interface, two backends.

    get_reasoner().generate_json(prompt, schema, max_new_tokens=None) -> dict
    get_reasoner().generate_text(prompt, max_new_tokens=128)          -> str   (C7 rationales)
    get_reasoner().generate_json_batch([(prompt, schema, mnt), ...])  -> [dict]  (speed)

Backend selected by c0_config.REASONER_BACKEND:
  * "transformers" — in-process FP16 + guided decoding (xgrammar/lmfe/outlines).
                     Strongest determinism (greedy on fixed hw); slow (no batching).
  * "vllm"         — OpenAI-compatible HTTP; paged attention + server-side batching.
                     FP16 (same weights/precision as transformers -> numbers stay
                     consistent). guided_json enforces the SAME schemas/enums.

Both log every generation to logs/c_reasoner_calls.jsonl (B6 determinism ethos;
future-proofs the C-fix list). The guided object must stay free of unbounded
strings — xgrammar (in-proc AND in vLLM) does not enforce string maxLength.
"""
from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c0_config as C  # noqa: E402


def _log_call(kind, backend, prompt, schema, max_new_tokens, output):
    try:
        C.REASONER_CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "backend": backend, "kind": kind,
               "n_prompt_chars": len(prompt or ""),
               "schema_keys": sorted((schema or {}).get("properties", {}).keys()) or None,
               "max_new_tokens": max_new_tokens, "output": output}
        with open(C.REASONER_CALL_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # logging must never break a run
        pass


# =====================================================================
# transformers backend (in-process, FP16)
# =====================================================================
class TransformersReasoner:
    backend = "transformers"

    def __init__(self, model_repo=None, dtype=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        self.model_repo = model_repo or C.REASONER_MODEL_REPO
        dtype = dtype or C.REASONER_DTYPE
        set_seed(C.REASONER_SEED)
        print(f"[reasoner] loading {self.model_repo} ({dtype}, device_map={C.REASONER_DEVICE_MAP}) …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_repo, dtype=getattr(torch, dtype), device_map=C.REASONER_DEVICE_MAP)
        self.model.eval()
        self.backend_lib = C.GUIDED_BACKEND
        self._set_seed = set_seed

    def _chat(self, prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def generate_json(self, prompt, json_schema, max_new_tokens=None):
        self._set_seed(C.REASONER_SEED)
        mnt = max_new_tokens or C.REASONER_MAX_NEW_TOKENS
        text = self._chat(prompt)
        if self.backend_lib == "xgrammar":
            raw = self._gen_xgrammar(text, json_schema, mnt)
        elif self.backend_lib == "lmfe":
            raw = self._gen_lmfe(text, json_schema, mnt)
        elif self.backend_lib == "outlines":
            obj = self._gen_outlines(text, json_schema, mnt)
            _log_call("json", self.backend, prompt, json_schema, mnt, obj)
            return obj
        else:
            raise ValueError(f"unknown GUIDED_BACKEND={self.backend_lib!r}")
        _log_call("json", self.backend, prompt, json_schema, mnt, raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"guided JSON truncated/invalid ({e}). Raise max_new_tokens, or "
                             f"lower REASON_TOP_K. First 200 chars: {raw[:200]!r}") from e

    def generate_text(self, prompt, max_new_tokens=128):
        import torch
        self._set_seed(C.REASONER_SEED)
        inputs = self.tokenizer(self._chat(prompt), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                      max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        txt = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        _log_call("text", self.backend, prompt, None, max_new_tokens, txt)
        return txt

    def generate_json_batch(self, items):
        # transformers has no server batching -> sequential (same behavior as before)
        return [self.generate_json(p, s, m) for (p, s, m) in items]

    def _gen_lmfe(self, text, json_schema, max_new_tokens):
        import torch
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer, JsonSchemaParser(json_schema))
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, prefix_allowed_tokens_fn=prefix_fn,
                                      do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    def _gen_xgrammar(self, text, json_schema, max_new_tokens):
        import torch
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        if getattr(self, "_xgr", None) is None:
            ti = xgr.TokenizerInfo.from_huggingface(self.tokenizer,
                                                    vocab_size=self.model.config.vocab_size)
            self._xgr = xgr.GrammarCompiler(ti)
        compiled = self._xgr.compile_json_schema(json.dumps(json_schema))
        proc = LogitsProcessor(compiled)                 # stateful: fresh per call
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, logits_processor=[proc], do_sample=False,
                                      num_beams=1, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    def _gen_outlines(self, text, json_schema, max_new_tokens):
        import outlines
        model = outlines.models.Transformers(self.model, self.tokenizer)
        gen = outlines.generate.json(model, json.dumps(json_schema))
        return gen(text, max_tokens=max_new_tokens, seed=C.REASONER_SEED)


# =====================================================================
# vLLM backend (OpenAI-compatible HTTP; FP16; server-side batching)
# =====================================================================
class VLLMReasoner:
    backend = "vllm"

    def __init__(self):
        import requests
        self._requests = requests
        self.base = C.REASONER_BASE_URL.rstrip("/")
        self.model = C.REASONER_SERVED_NAME
        self.headers = {"Content-Type": "application/json",
                        "Authorization": f"Bearer {C.VLLM_API_KEY}"}
        self.timeout = C.REASONER_HTTP_TIMEOUT
        self._wait_healthy()

    def _wait_healthy(self, timeout_s=600):
        url = self.base.replace("/v1", "") + "/health"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                if self._requests.get(url, timeout=5).status_code == 200:
                    print(f"[reasoner] vLLM healthy at {self.base} (model={self.model})")
                    return
            except self._requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not healthy at {url} after {timeout_s}s")

    def _post(self, body):
        r = self._requests.post(f"{self.base}/chat/completions",
                                headers=self.headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def generate_json(self, prompt, json_schema, max_new_tokens=None):
        mnt = max_new_tokens or C.REASONER_MAX_NEW_TOKENS
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "top_p": 1.0, "seed": C.REASONER_SEED, "max_tokens": mnt,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "decision", "schema": json_schema}}}
        raw = self._post(body)
        _log_call("json", self.backend, prompt, json_schema, mnt, raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"guided JSON truncated/invalid ({e}). "
                             f"First 200 chars: {raw[:200]!r}") from e

    def generate_text(self, prompt, max_new_tokens=128):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "top_p": 1.0, "seed": C.REASONER_SEED,
                "max_tokens": max_new_tokens}
        txt = self._post(body).strip()
        _log_call("text", self.backend, prompt, None, max_new_tokens, txt)
        return txt

    def generate_json_batch(self, items):
        # fire concurrently -> vLLM batches server-side. Order preserved.
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=C.REASONER_BATCH_CONCURRENCY) as ex:
            futs = {ex.submit(self.generate_json, p, s, m): i
                    for i, (p, s, m) in enumerate(items)}
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results


# =====================================================================
_REASONER = None

def get_reasoner():
    global _REASONER
    if _REASONER is None:
        _REASONER = VLLMReasoner() if C.REASONER_BACKEND == "vllm" else TransformersReasoner()
    return _REASONER


# back-compat alias (older code referenced `Reasoner`)
Reasoner = TransformersReasoner

if __name__ == "__main__":
    retrieved = ["p_9f2a1b3c4d", "p_1122334455"]
    schema = {"type": "object", "properties": {
        "cited_path_ids": {"type": "array", "maxItems": 3,
                           "items": {"type": "string", "enum": retrieved}},
        "confidence": {"type": "number"}, "supported": {"type": "boolean"}},
        "required": ["cited_path_ids", "confidence", "supported"]}
    r = get_reasoner()
    print(f"backend = {r.backend}")
    print(json.dumps(r.generate_json(
        "Cite path p_9f2a1b3c4d, supported=true, confidence 0.7. Reply as JSON.", schema), indent=2))

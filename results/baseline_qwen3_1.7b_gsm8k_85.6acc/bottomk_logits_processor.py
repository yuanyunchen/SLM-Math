"""
bottomk_logits_processor.py

Hard-constrained logits processor for a *fixed* bottom-k vocabulary.

Unlike the version that "recomputes bottom-k every step," this version works as follows:

    - Precompute a fixed bottom-k vocab S for a model (e.g., size=2000).
    - During generation, after each logits computation:
        - Keep the original logits only for tokens in S.
        - Set logits for tokens not in S to -inf.
    - Greedy or sampling then becomes "generate only within the fixed bottom-k."

This fits the fingerprint setting:
    1) On the base model, define a fixed bottom-k vocab (fingerprint space).
    2) The suspect model greedily generates y within its fixed bottom-k.
    3) Finally, check how many tokens in y belong to the base model's bottom-k vocab.
"""

from typing import Iterable, List, Optional

import torch
from transformers import LogitsProcessor


class BottomKLogitsProcessor(LogitsProcessor):
    """
    Hard-constrained logits processor for a *fixed* allowed vocab set.

    Pass allowed_token_ids at init (e.g., a model's bottom-2000 token id list).
    At each generation step:

        new_scores = -inf
        new_scores[..., allowed_token_ids] = original scores[..., allowed_token_ids]

    This forces softmax/greedy/sampling to operate only on allowed_token_ids.

    Args:
        allowed_token_ids: token ids allowed to be produced (e.g., bottom-k vocab).
    """

    def __init__(self, allowed_token_ids: Iterable[int]):
        allowed_token_ids = list(allowed_token_ids)
        if len(allowed_token_ids) == 0:
            raise ValueError("`allowed_token_ids` must be a non-empty list of token ids.")

        # Store as a long tensor for later scatter/index operations
        self.allowed_token_ids_tensor = torch.tensor(allowed_token_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: (batch_size, seq_len) current generated token sequence (unused here).
            scores:    (batch_size, vocab_size) logits at the current step.

        Returns:
            new_scores: (batch_size, vocab_size) with everything outside allowed_token_ids set to -inf.
        """
        if scores.ndim != 2:
            raise ValueError(
                f"`scores` is expected to be 2D (batch_size, vocab_size), got shape {scores.shape}"
            )

        batch_size, vocab_size = scores.shape

        # Move allowed ids to the same device as scores
        allowed_ids = self.allowed_token_ids_tensor.to(scores.device)

        # Guard: if vocab_size is smaller than expected, truncate (rare)
        allowed_ids = allowed_ids[allowed_ids < vocab_size]
        if allowed_ids.numel() == 0:
            raise ValueError(
                "After filtering with vocab_size, `allowed_token_ids` became empty. "
                f"vocab_size={vocab_size}."
            )

        # Set everything to -inf first
        new_scores = scores.new_full(scores.shape, float("-inf"))
        # Then copy logits at allowed_ids from the original scores
        # gather scores[..., allowed_ids] shape is (batch_size, len(allowed_ids))
        selected_scores = scores.index_select(dim=-1, index=allowed_ids)
        new_scores.scatter_(-1, allowed_ids.unsqueeze(0).expand(batch_size, -1), selected_scores)

        return new_scores


# ============================
# Helper: compute a "global bottom-k vocab" for a model (very rough version)
# Replace with a more robust statistic if needed.
# ============================

def compute_bottomk_vocab_for_model(
    model,
    tokenizer,
    k: int = 2000,
    device: Optional[str] = None,
    prompt: Optional[str] = None,
) -> List[int]:
    """
    Roughly compute a bottom-k vocab for a causal LM as a fingerprint space.

    Simplified idea:
        - Run one forward pass with a fixed prompt (or BOS).
        - Take the logits at the last position: (vocab_size,).
        - Sort logits ascending and take the first k token ids as bottom-k.

    This is crude but workable: a true "global bottom-k" could average over more prompts,
    but it's good enough as a starting experiment.

    Args:
        model:      HF AutoModelForCausalLM
        tokenizer:  corresponding tokenizer
        k:          bottom-k size, e.g., 2000
        device:     "cuda" / "mps" / "cpu"; if None, auto-detect from model.device
        prompt:     optional context; if None, use tokenizer.bos_token or a simple placeholder.

    Returns:
        bottomk_ids: token id list of length k
    """
    model.eval()

    if device is None:
        # Try inferring the device from model parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    if isinstance(device, torch.device):
        device = device.type

    # Prepare a simple prompt
    if prompt is None:
        if tokenizer.bos_token is not None:
            prompt = tokenizer.bos_token
        else:
            prompt = "Fingerprint base prompt."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Take logits at the last position: (batch_size=1, seq_len, vocab_size) -> (vocab_size,)
        logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    vocab_size = logits.shape[0]
    k = min(k, vocab_size)

    # Take bottom-k: the k tokens with the lowest logits
    _, bottomk_indices = torch.topk(logits, k=k, largest=False)

    bottomk_ids = bottomk_indices.tolist()
    return bottomk_ids


# ============================
# Demo: how to use (adjust for your project)
# ============================
if __name__ == "__main__":
    """
    Demo flow (illustrative):

    1. Choose a base model and compute its bottom-k vocab: base_bottomk_ids.
    2. Choose a suspect model and compute its bottom-k vocab: suspect_bottomk_ids.
       (If you want it to "greedy within its fixed bottom-k," use this set.)
    3. For suspect model generate:
        - Use BottomKLogitsProcessor(allowed_token_ids=suspect_bottomk_ids).
        - do_sample=False (greedy) or True (sampling within its bottom-k).
    4. After generating y_suspect, count tokens:
        - How many token ids are in base_bottomk_ids.
    """

    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

    # Demo only: in practice you'll use Qwen / TinyLlama etc.
    base_name = "gpt2"
    suspect_name = "gpt2"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {base_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_name).to(device)

    print("Computing base model bottom-k ids ...")
    base_bottomk_ids = compute_bottomk_vocab_for_model(
        base_model,
        base_tokenizer,
        k=2000,
        device=device,
    )
    print(f"Base model bottom-k size = {len(base_bottomk_ids)}")

    print(f"\nLoading suspect model: {suspect_name}")
    suspect_tokenizer = AutoTokenizer.from_pretrained(suspect_name)
    suspect_model = AutoModelForCausalLM.from_pretrained(suspect_name).to(device)

    # A simple fingerprint prompt; in practice use your constructed x'
    fingerprint_prompt = "This is a fingerprint prompt: "

    inputs = suspect_tokenizer(fingerprint_prompt, return_tensors="pt").to(device)

    # If you want the suspect to generate within its own bottom-k, compute it first:
    suspect_bottomk_ids = compute_bottomk_vocab_for_model(
        suspect_model,
        suspect_tokenizer,
        k=2000,
        device=device,
    )

    # Then use suspect_bottomk_ids as the allowed set:
    logits_processors = LogitsProcessorList(
        [
            BottomKLogitsProcessor(allowed_token_ids=suspect_bottomk_ids),
        ]
    )

    with torch.no_grad():
        outputs = suspect_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # greedy in its fixed bottom-k
            logits_processor=logits_processors,
        )

    generated_ids = outputs[0]
    generated_text = suspect_tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n=== Suspect model generated text (greedy in its bottom-k) ===\n")
    print(generated_text)

    # Stats: how many token ids in the generated sequence are in base_bottomk_ids
    base_bottomk_set = set(base_bottomk_ids)
    overlap_count = sum(int(t.item() in base_bottomk_set) for t in generated_ids)
    overlap_ratio = overlap_count / len(generated_ids)
    print(f"\nOverlap with base bottom-k vocab: {overlap_count}/{len(generated_ids)} "
          f"({overlap_ratio:.4f})")

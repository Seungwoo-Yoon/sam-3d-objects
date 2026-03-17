"""QLoRA quantization and gradient checkpointing utilities."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _make_linear4bit(linear: nn.Linear, quant_type: str, compute_dtype, double_quant: bool):
    """Return a bitsandbytes Linear4bit copy of an nn.Linear layer."""
    import bitsandbytes as bnb

    q = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        quant_type=quant_type,
        compute_dtype=compute_dtype,
        compress_statistics=double_quant,
    )
    # Params4bit stores the fp16 weight; actual quantization triggers on first forward
    q.weight = bnb.nn.Params4bit(
        linear.weight.data.cpu(),
        requires_grad=False,
        quant_type=quant_type,
    )
    if linear.bias is not None:
        q.bias = nn.Parameter(linear.bias.data)
    return q


def quantize_backbone_4bit(
    backbone: nn.Module,
    quant_type: str = "nf4",
    compute_dtype=None,
    double_quant: bool = True,
) -> nn.Module:
    """
    Quantize all nn.Linear weights inside *backbone* to 4-bit NF4 in-place.

    Works on both plain backbones and PEFT PeftModel wrappers:
      - plain nn.Linear            →  bnb.nn.Linear4bit
      - peft lora.Linear.base_layer →  bnb.nn.Linear4bit  (LoRA adapters stay fp16)

    Call this AFTER apply_lora() + loading LoRA checkpoint, but BEFORE
    moving the model back to GPU (or call .to(device) after to trigger quant).
    Then call peft.prepare_model_for_kbit_training() on the backbone.
    """
    import bitsandbytes as bnb
    from peft.tuners.lora.layer import Linear as LoraLinear

    if compute_dtype is None:
        compute_dtype = torch.bfloat16

    def _recurse(module: nn.Module):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoraLinear):
                # Quantize only the frozen base weight; LoRA A/B stay fp16
                base = child.base_layer
                if isinstance(base, nn.Linear) and not isinstance(base, bnb.nn.Linear4bit):
                    q = _make_linear4bit(base, quant_type, compute_dtype, double_quant)
                    child.base_layer = q
                    child._modules["base_layer"] = q
                # Do NOT recurse further — lora_A/B are nn.Linear but must stay fp16
            elif isinstance(child, nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
                setattr(module, child_name,
                        _make_linear4bit(child, quant_type, compute_dtype, double_quant))
            else:
                _recurse(child)

    _recurse(backbone)
    logger.info(
        "Quantized backbone: nn.Linear → bnb.nn.Linear4bit "
        "(quant_type=%s, double_quant=%s, compute_dtype=%s)",
        quant_type, double_quant, compute_dtype,
    )
    return backbone


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing on all transformer blocks inside the backbone.

    MidiSparseStructureFlowModel passes use_checkpoint down to every
    InteractionModule (and other blocks) that use it.  Setting use_checkpoint=True
    causes torch.utils.checkpoint.checkpoint to be called inside each block's
    forward(), so intermediate activations are NOT retained — they are recomputed
    during backward instead.  This trades ~2× more compute for a large reduction
    in peak activation memory.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "use_checkpoint") and not module.use_checkpoint:
            module.use_checkpoint = True
            count += 1
    logger.info("Gradient checkpointing enabled on %d module(s).", count)

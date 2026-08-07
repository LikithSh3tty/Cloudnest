"""Export the fine-tuned checkpoint to the serving format.

    python finetune/export_onnx.py

Serving is int8 ONNX, so the export quantizes to match. This is not a
formality: fine-tuning happens in fp32 and quantization can erase a small
gain entirely, which is why eval_retrieval.py is run against the int8
artifact rather than the checkpoint.

The output must expose the same graph shape backend/embed.py expects:
input_ids and attention_mask in, per-token hidden states out. embed() does
its own mask-weighted mean pooling, so the export must NOT bake in a
pooling layer - only the transformer body is exported, never the full
SentenceTransformer (which would append a Pooling module to the graph).
"""
import json
import shutil
from pathlib import Path

FINETUNE = Path(__file__).resolve().parent
CHECKPOINT = FINETUNE / "artifacts" / "finetuned-fp32"
ONNX_DIR = FINETUNE / "artifacts" / "finetuned-onnx"
BACKEND_MODEL_DIR = FINETUNE.parent / "backend" / "model"
TARGET_NAME = "model_finetuned_quint8_avx2.onnx"


def _find_transformer_body(checkpoint: Path) -> Path:
    """Locate the transformer-body subdirectory inside a saved
    SentenceTransformer checkpoint.

    Older sentence-transformers versions always used `0_Transformer/`. The
    installed version's on-disk layout was verified directly rather than
    assumed: this reads modules.json (the file SentenceTransformer.save()
    writes to record each module's type and path) and picks the entry whose
    type is the Transformer module, falling back to the conventional
    `0_Transformer/` name and finally the checkpoint root itself.
    """
    modules_json = checkpoint / "modules.json"
    if modules_json.exists():
        modules = json.loads(modules_json.read_text(encoding="utf-8"))
        for module in modules:
            if "Transformer" in module.get("type", ""):
                path = module.get("path", "")
                candidate = checkpoint / path if path else checkpoint
                if (candidate / "config.json").exists():
                    return candidate

    conventional = checkpoint / "0_Transformer"
    if conventional.exists():
        return conventional

    return checkpoint


def export(checkpoint: Path = CHECKPOINT) -> Path:
    """Export to ONNX, quantize to int8, install into backend/model/."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    source = _find_transformer_body(checkpoint)
    print(f"exporting transformer body from {source}")

    model = ORTModelForFeatureExtraction.from_pretrained(source, export=True)
    model.save_pretrained(ONNX_DIR)

    quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
    config = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=ONNX_DIR, quantization_config=config)

    quantized = next(ONNX_DIR.glob("*quantized*.onnx"))
    target = BACKEND_MODEL_DIR / TARGET_NAME
    shutil.copy(quantized, target)
    print(f"installed int8 model -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


if __name__ == "__main__":
    export()

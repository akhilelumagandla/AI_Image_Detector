"""
app.py
Two-stage Gradio deployment:
  1. Binary detector predicts Real vs AI-generated.
  2. If predicted AI-generated, the multiclass generator-ID model guesses
     which of the 8 generators likely produced it.

Run locally:   python app.py
Deploy:        push this file + checkpoints/ + requirements.txt to a
               Hugging Face Space (Gradio SDK). Use git-lfs for the
               checkpoint files (~45MB each).
NOTE: `import spaces` MUST be the first import, before torch, or
ZeroGPU's CUDA patching doesn't apply correctly. This only matters on
Hugging Face Spaces with ZeroGPU hardware -- the `spaces` package is a
no-op on any other environment (e.g. running this locally), so
`@spaces.GPU` is safe to leave in even when running on your own CUDA
machine or CPU-only.
"""
 
import spaces

import torch
import torch.nn.functional as F
import gradio as gr

import config
from model import build_model
from dataset import get_transforms


device = config.DEVICE
transform = get_transforms(train=False)  # clean resize+normalize -- same pipeline for both models

print(f"Loading binary detector from {config.CHECKPOINT_PATH} ...")
binary_model = build_model(backbone=config.BACKBONE, num_classes=config.NUM_CLASSES).to(device)
binary_model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
binary_model.eval()

print(f"Loading generator-ID model from {config.MULTICLASS_CHECKPOINT_PATH} ...")
multiclass_model = build_model(
    backbone=config.BACKBONE, num_classes=config.NUM_CLASSES_MULTICLASS
).to(device)
multiclass_model.load_state_dict(
    torch.load(config.MULTICLASS_CHECKPOINT_PATH, map_location=device)
)
multiclass_model.eval()

BINARY_LABELS = {0: "Real", 1: "AI-Generated"}
# Valid generator-guess classes only: exclude Real (0) -- not a valid answer
# once stage 1 says "fake" -- and SD14 (5), which has zero training samples
# in this dataset (see project notes / README).
GENERATOR_CLASSES = [cid for cid in config.GENERATOR_ID_TO_NAME if cid not in (0, 5)]


@spaces.GPU
def predict(image):
    if image is None:
        return {}, {}
 
    with torch.no_grad():
        img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
 
        # Stage 1: real vs fake
        binary_logits = binary_model(img_tensor)
        binary_probs = F.softmax(binary_logits, dim=1)[0].cpu()
        binary_result = {BINARY_LABELS[i]: float(binary_probs[i]) for i in range(2)}
 
        is_fake = binary_probs.argmax().item() == 1
        if not is_fake:
            return binary_result, {}
 
        # Stage 2: which generator (only runs if stage 1 said "fake")
        gen_logits = multiclass_model(img_tensor)[0]
        masked_logits = gen_logits.clone()
        masked_logits[0] = float("-inf")   # Real -- invalid here
        masked_logits[5] = float("-inf")   # SD14 -- untrained, zero samples
        gen_probs = F.softmax(masked_logits, dim=0).cpu()
 
        gen_result = {
            config.GENERATOR_ID_TO_NAME[cid]: float(gen_probs[cid])
            for cid in GENERATOR_CLASSES
        }
        return binary_result, gen_result
 


with gr.Blocks(title="AI Image Detector") as demo:
    gr.Markdown(
        "# AI-Generated Image Detector\n"
        "Upload an image to check whether it's real or AI-generated. "
        "If flagged as AI-generated, a second model guesses which generator "
        "likely produced it.\n\n"
        "**Trained on:** [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) "
        "-- ADM, BigGAN, GLIDE, Stable Diffusion 1.4/1.5, Wukong (2021-2022 era generators), "
        "plus limited testing on Midjourney and VQDM.\n\n"
        "**Known limitation:** this model is not trained on and will not reliably detect images "
        "from modern generators released after this dataset was built -- e.g. DALL-E 3, "
        "Midjourney v6, Flux, SDXL, or Bing/Copilot Image Creator. Cross-generator generalization "
        "to *any* unseen generator is an open problem (documented in the project README with "
        "a full diagnostic breakdown), and it's substantially worse for generators this different "
        "in era and quality from the training data."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload image")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            binary_output = gr.Label(label="Real vs AI-Generated", num_top_classes=2)
            generator_output = gr.Label(
                label="Likely generator (if AI-generated)", num_top_classes=5
            )

    submit_btn.click(
        fn=predict, inputs=image_input, outputs=[binary_output, generator_output]
    )
    image_input.change(
        fn=predict, inputs=image_input, outputs=[binary_output, generator_output]
    )


if __name__ == "__main__":
    demo.launch()
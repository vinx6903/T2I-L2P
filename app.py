"""
Usage:
    python app.py
"""

import os
import subprocess
import threading
import torch
import gradio as gr
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig

# ================= Configuration =================

# 1. Main model path
MAIN_MODEL_PATH = "./models/train/L2P_Standard/model-1k-merge.safetensors"

# 2. Text Encoder paths
TEXT_ENCODER_PATHS = [
    "/path/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors",
]

# 3. Tokenizer path
TOKENIZER_PATH = "/path/Z-Image-Turbo/tokenizer"

# 4. Server configuration
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 23231

# ================= GPU detection & pipeline pool =================

def get_available_gpus():
    """Detect all available GPUs and return [(index, mem_used_MB, mem_total_MB), ...]"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 3:
                idx = int(parts[0].strip())
                mem_used = float(parts[1].strip())
                mem_total = float(parts[2].strip())
                gpu_info.append((idx, mem_used, mem_total))
        return gpu_info
    except Exception as e:
        print(f"GPU detection failed ({e}), falling back to GPU 0 only")
        return [(0, 0, 0)]


def select_gpus_for_loading(num_gpus=8):
    """
    Select the list of GPUs used for loading models:
    1. Prefer idle GPUs (memory used < 100MB)
    2. If not enough, fill in with the GPUs that use the least memory
    3. Select at most num_gpus
    """
    all_gpus = get_available_gpus()
    if not all_gpus:
        return [0]

    free = [g for g in all_gpus if g[1] < 100]
    occupied = sorted([g for g in all_gpus if g[1] >= 100], key=lambda x: x[1])

    # Use idle cards first
    selected = free[:num_gpus]
    # If not enough, fill with the least-occupied cards
    remaining = num_gpus - len(selected)
    if remaining > 0:
        selected.extend(occupied[:remaining])

    return [g[0] for g in selected]


# Pick the list of GPUs to load on
gpu_ids = select_gpus_for_loading(8)
print(f"Will load pipelines on GPUs: {gpu_ids}")

# Load a pipeline on each selected GPU
pipelines = {}  # gpu_id -> pipeline
gpu_locks = {}  # gpu_id -> threading.Lock (prevent concurrent inference on the same GPU)

for gid in gpu_ids:
    dev = f"cuda:{gid}"
    print(f"Loading pipeline on {dev}...")
    p = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=dev,
        model_configs=[
            ModelConfig(path=[MAIN_MODEL_PATH]),
            ModelConfig(path=TEXT_ENCODER_PATHS),
        ],
        tokenizer_config=ModelConfig(path=TOKENIZER_PATH),
    )
    pipelines[gid] = p
    gpu_locks[gid] = threading.Lock()
    print(f"  GPU {gid}: pipeline loaded.")

print(f"All pipelines ready on GPUs: {list(pipelines.keys())}")


def acquire_pipeline():
    """
    Acquire an available (gpu_id, pipeline, lock).
    Prefer GPUs that are not currently locked (i.e. idle);
    if all are busy, wait for the GPU using the least memory to release its lock.
    """
    # 1. Try non-blocking acquisition of an idle GPU
    for gid in gpu_ids:
        if gpu_locks[gid].acquire(blocking=False):
            return gid, pipelines[gid], gpu_locks[gid]

    # 2. All busy: pick the one with the least memory used and wait
    gpus = get_available_gpus()
    gpus_on = [g for g in gpus if g[0] in gpu_ids]
    if gpus_on:
        best_gid = min(gpus_on, key=lambda x: x[1])[0]
    else:
        best_gid = gpu_ids[0]

    gpu_locks[best_gid].acquire(blocking=True)
    return best_gid, pipelines[best_gid], gpu_locks[best_gid]


# ================= Inference function =================

@torch.no_grad()
def generate(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    cfg_scale: float,
    height: int,
    width: int,
    seed: int,
):
    """Generate an image, automatically dispatching to an idle GPU"""
    if not prompt.strip():
        return None

    gid, pipe, lock = acquire_pipeline()
    dev = f"cuda:{gid}"
    try:
        print(f"[Request] Using GPU {gid} | prompt: {prompt[:60]}...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale,
            height=height,
            width=width,
            seed=seed,
            rand_device=dev,
            num_inference_steps=num_inference_steps,
        )
        print(f"[Request] GPU {gid} done.")
        return image
    except Exception as e:
        print(f"[Request] GPU {gid} error: {e}")
        raise e
    finally:
        lock.release()


# ================= Gradio UI =================

with gr.Blocks(title="L2P") as demo:
    gr.Markdown("# End-to-End Pixel Diffusion Model")

    with gr.Row():
        # Left: parameters panel
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="an origami pig on fire in the middle of a dark room with a pentagram on the floor.",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt(Optional)",
                value="",
                lines=2,
            )
            num_inference_steps = gr.Slider(
                minimum=1, maximum=100, step=1,
                label="Inference Steps", value=30,
            )
            cfg_scale = gr.Slider(
                minimum=0.1, maximum=10.0, step=0.1,
                label="CFG Scale", value=2.0,
            )
            height = gr.Slider(
                minimum=256, maximum=1024, step=16,
                label="Image Height", value=1024,
            )
            width = gr.Slider(
                minimum=256, maximum=1024, step=16,
                label="Image Width", value=1024,
            )
            seed = gr.Slider(
                minimum=0, maximum=1000000, step=1,
                label="Seed", value=42,
            )
            btn = gr.Button("Generate", variant="primary")

        # Right: output panel
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    btn.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            num_inference_steps,
            cfg_scale,
            height,
            width,
            seed,
        ],
        outputs=[output_image],
        concurrency_limit=8
    )

    # Example prompts
    gr.Examples(
        examples=[
            ["一位年轻汉族女性身穿红白相间、饰有金色花卉刺绣的汉服，背后远景是故宫。乌黑长发挽成精致发髻，点缀珍珠发簪与金饰。她面带温暖微笑望向镜头，双手举起纸卷，上面用毛笔写着\u201c你好\u201d。她身后是朱红色宫墙与金黄色琉璃瓦顶,传统红灯笼悬挂在石柱廊下,日光柔和。光滑的青石地面微微反光，增添空间层次。浅景深确保人物面部与文字清晰对焦。", "", 30, 2.0, 1024, 1024, 42],
            ["Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.", "", 30, 2.0, 1024, 1024, 42],
            ["an origami pig on fire in the middle of a dark room with a pentagram on the floor", "", 30, 2.0, 1024, 1024, 42],
        ],
        inputs=[
            prompt, negative_prompt, num_inference_steps,
            cfg_scale, height, width, seed,
        ],
        label="Prompt Examples",
    )

demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=True)

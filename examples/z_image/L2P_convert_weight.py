import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
import os
import argparse
from typing import List, Tuple
from torch.nn import RMSNorm

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


try:
    from diffsynth.models.z_image_dit_L2P import ZImageTransformerBlock, TimestepEmbedder, RopeEmbedder, FinalLayer
    # from diffsynth.models.utils import gradient_checkpoint_forward
    # from torch.nn.utils.rnn import pad_sequence
except ImportError:
    print("⚠️ Warning: Failed to import base modules from diffsynth.")
    pass

class MicroDiffusionModel(nn.Module):
    def __init__(self, in_channels, si_t_hidden_size):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2) 

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2) 
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.pool4 = nn.MaxPool2d(2, stride=2)  

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + si_t_hidden_size, 512, kernel_size=1), 
            nn.SiLU(),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)


    def forward(self, x, c):

        # U-Net Encoder
        enc1_out = self.enc1(x)                         # [B, 64, 16, 16]
        p1_out = self.pool1(enc1_out)                   # [B, 64, 8, 8]
        
        enc2_out = self.enc2(p1_out)                    # [B, 128, 8, 8]
        p2_out = self.pool2(enc2_out)                   # [B, 128, 4, 4]

        enc3_out = self.enc3(p2_out)                    # [B, 256, 4, 4]
        p3_out = self.pool3(enc3_out)                   # [B, 256, 2, 2]
        
        enc4_out = self.enc4(p3_out)                    # [B, 512, 2, 2]
        p4_out = self.pool4(enc4_out)                   # [B, 512, 1, 1]

        # Inject SiT feature into bottleneck
        bottleneck_input = torch.cat([p4_out, c], dim=1)
        bottleneck_out = self.bottleneck(bottleneck_input)   # [B, 512, 1, 1]

        # U-Net Decoder
        dec4_out = self.up4(bottleneck_out)                 # [B, 512, 2, 2]
        dec4_out = torch.cat([dec4_out, enc4_out], dim=1)   # Skip
        dec4_out = self.dec4(dec4_out)                      # [B, 256, 2, 2]

        dec3_out = self.up3(dec4_out)                       # [B, 256, 4, 4]
        dec3_out = torch.cat([dec3_out, enc3_out], dim=1)   # Skip
        dec3_out = self.dec3(dec3_out)                      # [B, 128, 4, 4]

        dec2_out = self.up2(dec3_out)                       # [B, 128, 8, 8]
        dec2_out = torch.cat([dec2_out, enc2_out], dim=1)   # Skip
        dec2_out = self.dec2(dec2_out)                      # [B, 64, 8, 8]

        dec1_out = self.up1(dec2_out)                       # [B, 64, 16, 16]
        dec1_out = torch.cat([dec1_out, enc1_out], dim=1)   # Skip
        dec1_out = self.dec1(dec1_out)                      # [B, 64, 16, 16]
           
        x_out = self.out_conv(dec1_out)                     # [B, C, 16, 16]
           
        return x_out


class ZImageDiT(nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

    def __init__(
        self,
        all_patch_size=(16,),
        all_f_patch_size=(1,),
        in_channels=3,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        # all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            # final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            # all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        # self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.local_decoder = MicroDiffusionModel(
            in_channels=in_channels,
            si_t_hidden_size=dim
        )


        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def forward(self, *args, **kwargs):
        pass 

# ==============================================================================
# 2. convert
# ==============================================================================

def convert_weights(latent_ckpt_files=None, output_path=None):
    # ---------------- Configuration ----------------
    if latent_ckpt_files is None:
        latent_ckpt_files = [
            "/path/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "/path//Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "/path/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors"
        ]
    if output_path is None:
        output_path = "pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors"
    
    pixel_config = dict(
        all_patch_size=(16,),
        all_f_patch_size=(1,),
        in_channels=3,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        cap_feat_dim=2560
    )
    # -----------------------------------------

    print(f"🚀 [DiffSynth] Starting weight conversion task...")

    # 1. Load Latent weights (merge shards)
    print(f"📂 Loading Latent weights ({len(latent_ckpt_files)} shards in total)...")
    source_state_dict = {}
    for f in latent_ckpt_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Weight file not found: {f}")
        print(f"   -> Reading: {f}")
        part_weights = load_file(f)
        source_state_dict.update(part_weights)
    print(f"✅ Latent weights loaded, {len(source_state_dict)} parameter tensors in total.")
    
    # 2. Initialize the Pixel model
    print(f"🔨 Initializing Pixel Space model (Patch=16, Ch=3)...")
    # Note: all weights here are randomly initialized
    model = ZImageDiT(**pixel_config)
    target_state_dict = model.state_dict()


   # 3. Smart migration
    print(f"🔄 Starting weight migration...")
    final_state_dict = {}
    
    transferred = 0
    skipped_shape = 0
    skipped_missing = 0

    for key, target_param in target_state_dict.items():
        # Case A: Key exists and shape matches -> copy (Backbone, Refiners, TimeEmbedder)
        if key in source_state_dict:
            source_param = source_state_dict[key]
            if source_param.shape == target_param.shape:
                final_state_dict[key] = source_param
                transferred += 1
            else:
                # Case B: Key exists but shape mismatches -> keep random initialization (Embedder)
                # e.g.: all_x_embedder.weight (Latent is 64->3840, Pixel is 768->3840)
                print(f"   ⚠️ [Shape mismatch] {key}: Latent {source_param.shape} -> Pixel {target_param.shape} (keeping random initialization)")
                final_state_dict[key] = target_param
                skipped_shape += 1
        else:
            # Case C: Key does not exist
            # Print to confirm
            if skipped_missing < 40: # only print the first 10 to avoid flooding logs
                print(f"   🆕 [New layer - random init] {key}")
            elif skipped_missing == 40:
                print(f"   ... (more new layers)")
            
            final_state_dict[key] = target_param
            skipped_missing += 1

    print("-" * 40)
    print(f"📊 Statistics:")
    print(f"   - Successfully reused weights: {transferred} layers (Transformer Backbone)")
    print(f"   - Reset due to shape conflict: {skipped_shape} layers")
    print(f"   - Newly added modules initialized: {skipped_missing} layers (Decoder & Embedder)")
    

    # ==============================================================================
    # 4. Full strict verification (newly added)
    # ==============================================================================
    print("\n" + "="*50)
    print("🔍 Performing full layer-wise verification...")
    
    stats = {
        "success_copy": 0,      # Same shape and exactly equal values
        "shape_mismatch": 0,    # Shape mismatch (expected)
        "new_layer": 0,         # Layer not present in Latent (expected)
        "copy_failed": 0        # ❌ Critical error: same shape but values differ
    }
    
    failed_layers = []

    for key, target_tensor in final_state_dict.items():
        # Case A: Key exists in source weights
        if key in source_state_dict:
            source_tensor = source_state_dict[key]
            
            # Check shape
            if source_tensor.shape == target_tensor.shape:
                # Core verification: compute difference
                # Note: floating-point may have tiny numerical errors, but direct copy is usually 0
                diff = (source_tensor - target_tensor).abs().sum().item()
                
                if diff == 0.0:
                    stats["success_copy"] += 1
                else:
                    stats["copy_failed"] += 1
                    failed_layers.append((key, diff))
            else:
                stats["shape_mismatch"] += 1
        
        # Case B: Key is a new layer
        else:
            stats["new_layer"] += 1

    # Print verification report
    print(f"📊 Verification report:")
    print(f"   ✅ Successfully reused (values exactly match): {stats['success_copy']} layers")
    print(f"   ⚠️ Shape conflict (keeping random initialization): {stats['shape_mismatch']} layers (e.g. embedder)")
    print(f"   🆕 Newly added modules (keeping random initialization): {stats['new_layer']} layers (e.g. decoder)")
    
    if stats["copy_failed"] > 0:
        print(f"\n❌ Critical warning: found {stats['copy_failed']} layers that should be copied but values do not match!")
        for name, diff in failed_layers:
            print(f"   - {name} (Diff: {diff})")
        raise RuntimeError("Weight copy verification failed, please check the code logic!")
    else:
        print(f"\n🎉 Perfect! All shape-matched layers ({stats['success_copy']}) have been migrated exactly.")
    print("="*50 + "\n")

    # 5. Save
    print(f"💾 Saving converted weights to: {output_path}")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_file(final_state_dict, output_path)
    print("✅ Task completed.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Z-Image-Turbo (Latent) transformer weights to Z-Image Pixel Init weights."
    )
    parser.add_argument(
        "--latent_ckpt_files",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of source Latent transformer safetensors shards. "
            "Pass multiple paths separated by spaces. "
            "If omitted, falls back to the default Z-Image-Turbo paths."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for the converted Pixel-Init safetensors file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_weights(
        latent_ckpt_files=args.latent_ckpt_files,
        output_path=args.output_path,
    )
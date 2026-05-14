import argparse
import os
import torch
from safetensors.torch import load_file, save_file
from diffsynth.core.loader import hash_model_file

def merge_safetensors(path_a, path_b, output_path):
    print(f"Loading file A: {path_a}")
    weights_a = load_file(path_a)
    
    print(f"Loading file B: {path_b}")
    weights_b = load_file(path_b)
    
    # Convert to a mutable dict (in case it is read-only)
    merged_weights = dict(weights_b)
    
    count = 0
    missing_keys = []
    
    print("Starting to overwrite weights...")
    for key, value in weights_a.items():
        if key in merged_weights:
            # Check whether shapes match (optional, but recommended just in case)
            if merged_weights[key].shape == value.shape:
                merged_weights[key] = value
                count += 1
            else:
                print(f"Warning: Key '{key}' has mismatched shapes! A: {value.shape}, B: {merged_weights[key].shape}")
        else:
            missing_keys.append(key)
            
    if missing_keys:
        print(f"Note: the following {len(missing_keys)} keys in A were not found in B and were skipped:")
        # print(missing_keys) # uncomment if you want to see exactly which keys were missing
    
    print(f"Successfully overwrote {count} weight parameters.")
    
    # Save result
    print(f"Saving to: {output_path}")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_file(merged_weights, output_path)
    print("Save completed!")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge two safetensors files: overwrite matching keys in B "
            "(superset / base model) with values from A (subset / new weights)."
        )
    )
    parser.add_argument(
        "--file_a",
        type=str,
        default="/path/step-xxx.safetensors",
        help="Path to file A (subset, new weights to merge in).",
    )
    parser.add_argument(
        "--file_b",
        type=str,
        default="./pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors",
        help="Path to file B (superset, base model).",
    )
    parser.add_argument(
        "--file_out",
        type=str,
        default="/path/model-merge.safetensors",
        help="Output path for the merged safetensors file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_safetensors(args.file_a, args.file_b, args.file_out)
    print(hash_model_file(args.file_out))

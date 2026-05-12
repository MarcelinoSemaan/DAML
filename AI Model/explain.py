#!/usr/bin/env python3
"""
Explain LSTM predictions with per-feature attribution using Integrated Gradients.
Usage: python explain.py safe.exe --model-dir ./artifacts --top 20
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, DeepLift

from main import EmberLSTM, TARGET_NF, file_to_features


def load_artifacts(artifacts_dir: Path):
    with open(artifacts_dir / "feat_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    with open(artifacts_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    n_cols = len(feat_cols)
    n_timesteps = max(1, int(np.ceil(n_cols / TARGET_NF)))
    device = torch.device("cpu")
    
    model = EmberLSTM(TARGET_NF, n_timesteps).to(device)
    model.load_state_dict(torch.load(artifacts_dir / "ember_lstm_best.pt", map_location=device))
    model.eval()
    
    return model, feat_cols, scaler, n_cols, n_timesteps, device


def extract_features(file_path, feat_cols, scaler):
    return file_to_features(file_path, feat_cols, scaler)


def integrated_gradients_explain(model, tensor, n_cols, n_steps=100, top_k=20):
    """
    Integrated Gradients: attribute each of the 64*timesteps input values
    by integrating gradients from baseline (zero) to input.
    """
    baseline = torch.zeros_like(tensor)
    
    # Forward that takes raw input and returns logit
    def forward_raw(x):
        proj = model.input_proj(x)           # (1, T, 128)
        lstm_out, _ = model.lstm(proj)        # (1, T, 512)
        attn_scores = model.attn(lstm_out).squeeze(-1)  # (1, T)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (1, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (1, 512)
        return model.classifier(context).squeeze(-1)    # (1,)
    
    ig = IntegratedGradients(forward_raw)
    
    # n_steps=100 gives good approximation; increase for precision
    attributions, delta = ig.attribute(
        tensor, 
        baselines=baseline,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    # Flatten: (1, T, 64) -> (n_features,)
    attr_np = attributions[0].cpu().numpy().flatten()[:n_cols]
    
    # Take absolute importance (sign tells direction, but for explanation we care about magnitude)
    attr_abs = np.abs(attr_np)
    
    # Get prediction
    with torch.no_grad():
        prob = torch.sigmoid(forward_raw(tensor)).item()
    
    # Top features by absolute attribution
    top_idx = np.argsort(attr_abs)[-top_k:][::-1]
    
    return prob, attr_np, attr_abs, top_idx, float(delta)


def deep_lift_explain(model, tensor, n_cols, top_k=20):
    """
    DeepLIFT: faster alternative to IG, compares to baseline.
    Good for ReLU/GELU networks.
    """
    baseline = torch.zeros_like(tensor)
    
    def forward_raw(x):
        proj = model.input_proj(x)
        lstm_out, _ = model.lstm(proj)
        attn_scores = model.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return model.classifier(context).squeeze(-1)
    
    dl = DeepLift(forward_raw)
    attributions = dl.attribute(tensor, baselines=baseline)
    
    attr_np = attributions[0].cpu().numpy().flatten()[:n_cols]
    attr_abs = np.abs(attr_np)
    
    with torch.no_grad():
        prob = torch.sigmoid(forward_raw(tensor)).item()
    
    top_idx = np.argsort(attr_abs)[-top_k:][::-1]
    
    return prob, attr_np, attr_abs, top_idx


def group_by_prefix(feat_name):
    """Map EMBER feature name to semantic group."""
    if feat_name.startswith("byteentropy."):
        return "byte_entropy"
    elif feat_name.startswith("histogram."):
        return "histogram"
    elif feat_name.startswith("section.") or feat_name.startswith("sections."):
        return "sections"
    elif feat_name.startswith("import.") or feat_name.startswith("imports."):
        return "imports"
    elif feat_name.startswith("export.") or feat_name.startswith("exports."):
        return "exports"
    elif feat_name.startswith("general."):
        return "general"
    elif feat_name.startswith("header.") or feat_name.startswith("optional_header."):
        return "header"
    elif feat_name.startswith("strings."):
        return "strings"
    elif feat_name.startswith("datadirectories."):
        return "data_directories"
    elif feat_name.startswith("authenticode.") or feat_name.startswith("signature."):
        return "authenticode"
    else:
        return "other"


def print_explanation(prob, attr_abs, attr_signed, feat_cols, top_idx, method_name):
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"Malicious probability: {prob:.4f}  ({'MALICIOUS' if prob >= 0.5 else 'BENIGN'})")
    print(f"{'='*60}")
    
    print(f"\nTop {len(top_idx)} features by importance:")
    print("-" * 70)
    print(f"{'Rank':>4} | {'Attribution':>11} | {'Feature Name'}")
    print("-" * 70)
    
    max_attr = attr_abs[top_idx[0]]
    for rank, idx in enumerate(top_idx, 1):
        feat = feat_cols[idx]
        val = attr_abs[idx]
        signed = attr_signed[idx]
        direction = "↑" if signed > 0 else "↓"  # ↑ pushes toward malware, ↓ toward benign
        
        # Scale bar relative to top feature
        bar_len = int(40 * val / max_attr) if max_attr > 0 else 0
        bar = "█" * bar_len
        
        print(f"{rank:>4} | {val:>10.6f} {direction} | {bar} {feat}")
    
    # Group-level aggregation
    print(f"\n{'='*60}")
    print("Feature group contribution (by absolute attribution):")
    print("-" * 70)
    
    group_scores = {}
    group_directional = {}
    for i, feat in enumerate(feat_cols):
        g = group_by_prefix(feat)
        group_scores[g] = group_scores.get(g, 0.0) + attr_abs[i]
        group_directional[g] = group_directional.get(g, 0.0) + attr_signed[i]
    
    total = sum(group_scores.values())
    sorted_groups = sorted(group_scores.items(), key=lambda x: -x[1])
    
    for group, score in sorted_groups:
        pct = score / total * 100
        net_dir = group_directional[group]
        direction_str = "pushes MALICIOUS" if net_dir > 0 else "pushes BENIGN"
        bar_len = int(30 * score / sorted_groups[0][1])
        bar = "█" * bar_len
        print(f"{pct:5.1f}% | {bar} {group:<20} ({direction_str})")
    
    # Show what would change the prediction
    print(f"\n{'='*60}")
    print("What would reduce the malicious score?")
    print("-" * 70)
    
    # Find features with negative attribution (push toward benign) that are currently high
    benign_pushers = [(i, attr_signed[i], feat_cols[i]) 
                      for i in range(len(feat_cols)) 
                      if attr_signed[i] < -0.001]
    benign_pushers.sort(key=lambda x: x[1])  # Most negative first
    
    if benign_pushers:
        for idx, val, feat in benign_pushers[:10]:
            print(f"  {val:>10.6f} ↓ | {feat}")
    else:
        print("  (No strong benign-pushing features detected in this sample)")


def compare_with_baseline(model, tensor, feat_cols, n_cols, baseline_tensor=None):
    """
    Compare against a known benign sample's feature vector as baseline
    instead of zero. This shows 'what makes this different from benign'.
    """
    if baseline_tensor is None:
        return None
    
    def forward_raw(x):
        proj = model.input_proj(x)
        lstm_out, _ = model.lstm(proj)
        attn_scores = model.attn(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return model.classifier(context).squeeze(-1)
    
    ig = IntegratedGradients(forward_raw)
    attr, _ = ig.attribute(tensor, baselines=baseline_tensor, n_steps=50, return_convergence_delta=True)
    
    attr_np = attr[0].cpu().numpy().flatten()[:n_cols]
    return attr_np


def main():
    parser = argparse.ArgumentParser(description="Explain LSTM malware predictions")
    parser.add_argument("file", help="PE file to analyze")
    parser.add_argument("--model-dir", default="./artifacts", help="Path to artifacts/")
    parser.add_argument("--top", type=int, default=20, help="Number of top features")
    parser.add_argument("--method", choices=["ig", "deeplift", "both"], default="both",
                       help="Attribution method")
    parser.add_argument("--baseline-file", default=None, 
                       help="Known benign file to use as baseline instead of zero")
    parser.add_argument("--steps", type=int, default=100, help="IG integration steps")
    args = parser.parse_args()
    
    model, feat_cols, scaler, n_cols, n_timesteps, device = load_artifacts(Path(args.model_dir))
    
    # Extract features from target file
    features_arr = extract_features(args.file, feat_cols, scaler)
    tensor = torch.tensor(features_arr, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Optional: load baseline for comparative explanation
    baseline_tensor = None
    if args.baseline_file:
        base_arr = extract_features(args.baseline_file, feat_cols, scaler)
        baseline_tensor = torch.tensor(base_arr, dtype=torch.float32).unsqueeze(0).to(device)
        print(f"Using baseline: {args.baseline_file}")
    
    if args.method in ("ig", "both"):
        print(f"\nComputing Integrated Gradients (n_steps={args.steps})...")
        prob, attr_signed, attr_abs, top_idx, delta = integrated_gradients_explain(
            model, tensor, n_cols, n_steps=args.steps, top_k=args.top
        )
        print(f"Convergence delta: {delta:.6f} (should be near zero)")
        print_explanation(prob, attr_abs, attr_signed, feat_cols, top_idx, 
                         "Integrated Gradients" + (" (vs baseline)" if baseline_tensor else " (vs zero)"))
    
    if args.method in ("deeplift", "both"):
        print(f"\nComputing DeepLIFT...")
        prob, attr_signed, attr_abs, top_idx = deep_lift_explain(
            model, tensor, n_cols, top_k=args.top
        )
        print_explanation(prob, attr_abs, attr_signed, feat_cols, top_idx, "DeepLIFT")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_member03.py -- Master Integrated Execution Workflow for Member 03
=============================================================================
Project   : Project 04 - Member 03 (Explainability, Validation & Integration)
Topic     : Neonatal Incubator Temperature and Oxygen Regulation (DDQN + SHAP)
Reference : Member 1 (Environment, Baseline), Member 2 (DDQN, Checkpoints)

Purpose:
--------
Unified master entry point integrating:
  1. Environment initialization and contract verification.
  2. Loading trained DDQN checkpoints (Member 2) without retraining.
  3. Multi-scenario empirical decision validation against baseline (Member 1).
  4. Mathematical SHAP explainability analysis (DeepSHAP & KernelSHAP).
  5. High-resolution visualization export and automated summary reporting.

Usage:
------
    python run_member03.py
    python run_member03.py --episodes-per-scenario 5 --checkpoint models/ddqn_tuned_seed0/best.pt

>>> SIMULATION-ONLY RESEARCH PROJECT - NOT FOR CLINICAL USE <<<
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict

# Environment configuration
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_CUR_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from shap_explainer import run_shap_analysis, prepare_explanation
from validation import run_decision_validation


def run_member03_workflow(
    checkpoint: str = "models/ddqn_tuned_seed0/best.pt",
    output_base_dir: str = "outputs/member03",
    episodes_per_scenario: int = 5,
    shap_method: str = "deep",
    device: str = "cpu",
    seed: int = 42,
    skip_shap: bool = False,
) -> Dict[str, Any]:
    """Execute complete end-to-end Member 03 pipeline."""
    start_time = time.time()
    print("=" * 80)
    print("  PROJECT 04 -- MEMBER 03: SHAP EXPLAINABILITY, VALIDATION & INTEGRATION")
    print("  Neonatal Incubator Temperature & Oxygen Regulation (DDQN + SHAP)")
    print("=" * 80)
    print(f"[*] Checkpoint      : {checkpoint}")
    print(f"[*] Output Base Dir : {output_base_dir}")
    print(f"[*] Episodes / Scn  : {episodes_per_scenario}")
    print(f"[*] Compute Device  : {device}")
    print(f"[*] Base Seed       : {seed}")
    print(f"[*] SHAP Method     : {shap_method} (skip={skip_shap})")
    print("=" * 80)

    context = None if skip_shap else prepare_explanation(checkpoint, device, seed, shap_method)

    # 1. Decision Validation Stage
    val_out_dir = os.path.join(output_base_dir, "validation")
    print("\n[Validation] Starting Multi-Scenario Decision Validation...")
    val_summary = run_decision_validation(
        checkpoint_path=checkpoint,
        output_dir=val_out_dir,
        episodes_per_scenario=episodes_per_scenario,
        device=device,
        seed_base=seed,
        shap_method=shap_method,
        skip_shap=skip_shap,
        context=context,
    )

    # 2. SHAP Explainability Stage
    shap_out_dir = os.path.join(output_base_dir, "shap")
    shap_manifest: Dict[str, Any] = {}
    if not skip_shap:
        print("\n[SHAP] Starting SHAP Explainability Engine...")
        shap_manifest = run_shap_analysis(
            checkpoint_path=checkpoint,
            output_dir=shap_out_dir,
            method=shap_method,
            device=device,
            seed=seed,
            context=context,
        )
    else:
        print("\n[SHAP] SHAP analysis skipped via flag.")

    elapsed = time.time() - start_time

    # 3. Final Summary Report
    print("\n" + "=" * 80)
    print("  MEMBER 03 WORKFLOW EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Completed in {elapsed:.2f} seconds.")
    print("\nComparative Results (DDQN vs Baseline):")
    print("-" * 80)
    print(f"{'Scenario':<34} | {'DDQN Return':<13} | {'Base Return':<13} | {'Advantage':<10}")
    print("-" * 80)

    for scn_name, scn_data in val_summary["by_scenario"].items():
        ddqn_ret = scn_data["ddqn"]["mean_total_reward"]
        base_ret = scn_data["baseline"]["mean_total_reward"]
        adv = scn_data["ddqn_advantage_return"]
        print(f"{scn_name:<34} | {ddqn_ret:>11.2f}  | {base_ret:>11.2f}  | {adv:>+9.2f}")

    print("-" * 80)
    overall_ddqn = val_summary["overall"]["ddqn"]["mean_total_reward"]
    overall_base = val_summary["overall"]["baseline"]["mean_total_reward"]
    net_adv = val_summary["overall"]["net_ddqn_improvement"]
    print(f"{'OVERALL AGGREGATE (All 5 Scenarios)':<34} | {overall_ddqn:>11.2f}  | {overall_base:>11.2f}  | {net_adv:>+9.2f}")
    print("=" * 80)

    print("\nGenerated Deliverables & Artifacts:")
    print(f"  [Validation CSVs] : {val_out_dir}/normal_scenario_results.csv")
    print(f"                     {val_out_dir}/abnormal_temperature_results.csv")
    print(f"                     {val_out_dir}/abnormal_oxygen_results.csv")
    print(f"                     {val_out_dir}/validation_episodes.csv")
    print(f"  [Validation JSON] : {val_out_dir}/validation_summary.json")
    if not skip_shap:
        print(f"  [SHAP Plots]      : {shap_out_dir}/shap_summary.png")
        print(f"                     {shap_out_dir}/shap_feature_importance.png")
        print(f"                     {shap_out_dir}/shap_policy_feature_importance.png")
        print(f"                     {shap_out_dir}/shap_waterfall.png")
        print(f"                     {shap_out_dir}/shap_action_contrast.png")
        print(f"                     {shap_out_dir}/shap_scenario_comparison.png")
        print(f"  [SHAP Manifest]   : {shap_out_dir}/shap_manifest.json")
        print(f"  [SHAP Background] : {shap_out_dir}/background_metadata.json")
    print("=" * 80)
    print("Validation completed." if skip_shap else "Validation and SHAP export completed.")
    print("=" * 80)

    return {
        "elapsed_seconds": elapsed,
        "validation_summary": val_summary,
        "shap_manifest": shap_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Integrated Workflow for Member 03 (DDQN + SHAP)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/ddqn_tuned_seed0/best.pt",
        help="Path to trained DDQN checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/member03",
        help="Base output directory for Member 03 results",
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=5,
        help="Number of evaluation episodes per scenario",
    )
    parser.add_argument(
        "--shap-method",
        type=str,
        choices=["deep", "kernel"],
        default="deep",
        help="SHAP explainer engine (deep or kernel)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device ('cpu' or 'gpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP analysis and run validation only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_member03_workflow(
        checkpoint=args.checkpoint,
        output_base_dir=args.output_dir,
        episodes_per_scenario=args.episodes_per_scenario,
        shap_method=args.shap_method,
        device=args.device,
        seed=args.seed,
        skip_shap=args.skip_shap,
    )

#!/bin/bash
# Run the proxy re-ID score for every already-measured generator, writing one log per tag.
# Runs sequentially so the GPU is not thrashed while the baseline retrain is in flight.
cd /workspace

run() {
    tag="$1"; shift
    echo "[$tag] proxy start $(date '+%H:%M:%S')"
    python proxy_reid.py "$@" > "/workspace/proxy_$tag.log" 2>&1
    echo "[$tag] proxy done   $(date '+%H:%M:%S') -> $(grep -oP 'PROXY_AUC: \K[0-9.]+' /workspace/proxy_$tag.log)"
}

run run_1    --checkpoint ./archive/train_prichexy_net_run_1/generator_lowest_total_loss.pth
run run_2    --checkpoint ./archive/train_prichexy_net_run_2_h1fix/generator_lowest_total_loss.pth
run run_3    --checkpoint ./archive/train_prichexy_net_run_3_entropy/generator_lowest_total_loss.pth
run run_4    --checkpoint ./archive/train_prichexy_net_run_4_ensemble/generator_lowest_total_loss.pth
run baseline_none
run randomwarp --stochastic_lambda 1.0 --checkpoint ./archive/train_prichexy_net_run_1/generator_lowest_total_loss.pth

echo "ALL PROXY RUNS COMPLETE $(date)"
#!/bin/bash
# run_mrt_sweep.sh — sweep λ values across all domains
# Usage: bash run_mrt_sweep.sh

# DOMAINS=("emea" "news_commentary" "opus_books")
DOMAINS=("emea")
LAMBDAS=(0.1 0.5 0.9)

for DOMAIN in "${DOMAINS[@]}"; do
    for LAM in "${LAMBDAS[@]}"; do
        echo "=========================================="
        echo "Domain: $DOMAIN | λ=$LAM"
        echo "=========================================="
        python src/mrt_train.py \
            --domain      "$DOMAIN" \
            --lambda_val  "$LAM"
    done
done

echo "All runs complete."

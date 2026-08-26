#!/usr/bin/env bash
# ============================================================================
# V2 pipeline queue: waits for the current GPU job to finish, then runs the
# full PriCheXy-Net V2 chain sequentially:
#   1. train anonymizer (attention U-Net, 250 epochs)
#   2. retrain SNN attacker against it (privacy AUC on VAL)
#   3. evaluate CheXNet classifier on anonymized images (utility, VAL fold)
#
# Safety properties:
#   * Never touches the watched process; only observes it.
#   * Requires BOTH conditions before starting: watched PID exited AND
#     >= MIN_FREE_MB free VRAM.
#   * Step N+1 runs only if step N succeeded (checkpoint existence is also
#     verified explicitly).
#   * All output logged to research_runs/V2_QUEUE/.
# ============================================================================

set -u

WORKDIR="/home/minhtt/Neups_workshop"
PY="$WORKDIR/.venv/bin/python"
WATCH_PID="2612818"          # phase_c_driver.py currently holding the GPU
MIN_FREE_MB=13000            # training needs ~9 GB peak + ~2 GB for the extra
                             # DenseNet forwards; keep a safety margin
POLL_SECONDS=300             # poll every 5 minutes

CFG_DIR="./config_files/"
TRAIN_CFG="config_anonymization_v2_attention_run1.json"
SNN_CFG="config_retrainSNN_v2_run1.json"
EVAL_CFG="config_eval_classifier_v2_run1.json"

TRAIN_CKPT="./archive/v2_attention_feat1_run1/generator_lowest_total_loss.pth"

LOGDIR="$WORKDIR/research_runs/V2_QUEUE"
STATUS_FILE="$LOGDIR/status.txt"
mkdir -p "$LOGDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/queue.log"
}

# Machine-readable status for downstream queues: RUNNING / COMPLETE / FAILED.
# A dedicated file avoids any ambiguity from grepping append-only logs that
# contain failures of PREVIOUS sessions.
set_status() {
    echo "$1" > "$STATUS_FILE"
}

log "=== V2 QUEUE STARTED (watching PID $WATCH_PID) ==="
set_status "RUNNING"

# ---------------------------------------------------------------------------
# Phase 0: wait for the current GPU job to exit.
# ---------------------------------------------------------------------------
while kill -0 "$WATCH_PID" 2>/dev/null; do
    log "PID $WATCH_PID still running; waiting ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
done
log "PID $WATCH_PID has exited."

# Extra safety margin so the previous job's CUDA context fully releases.
sleep 60

# ---------------------------------------------------------------------------
# Phase 1: wait for enough free VRAM (in case another job grabbed the GPU).
# ---------------------------------------------------------------------------
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    log "Free VRAM: ${FREE} MiB (need >= ${MIN_FREE_MB} MiB)"
    if [ "${FREE:-0}" -ge "$MIN_FREE_MB" ]; then
        break
    fi
    sleep "$POLL_SECONDS"
done
log "GPU is free. Starting V2 pipeline."

cd "$WORKDIR" || { log "FATAL: cannot cd to $WORKDIR"; exit 1; }

# ---------------------------------------------------------------------------
# train_architecture_v2.py refuses (os.mkdir, fail-closed) to reuse an
# existing experiment dir. A prior killed/crashed attempt leaves that dir
# behind, so every retry died instantly with exit 1 until someone deleted it
# by hand. Archive it out of the way instead of crashing.
# ---------------------------------------------------------------------------
EXP_DIR="$(dirname "$TRAIN_CKPT")"
if [ -d "$EXP_DIR" ]; then
    STAMP=$(date '+%Y%m%d_%H%M%S')
    log "Found pre-existing $EXP_DIR from a previous attempt; moving it to ${EXP_DIR}_preempted_${STAMP} before restarting."
    mv "$EXP_DIR" "${EXP_DIR}_preempted_${STAMP}"
fi

# ---------------------------------------------------------------------------
# Step 1: train the V2 anonymizer.
# ---------------------------------------------------------------------------
log "STEP 1/3: training V2 anonymizer ($TRAIN_CFG)"
if "$PY" train_architecture_v2.py \
        --config_path "$CFG_DIR" --config "$TRAIN_CFG" \
        > "$LOGDIR/train.log" 2>&1; then
    log "STEP 1 finished OK."
else
    CODE=$?
    log "STEP 1 FAILED (exit $CODE). See $LOGDIR/train.log -- stopping queue."
    set_status "FAILED"
    exit 1
fi

if [ ! -f "$TRAIN_CKPT" ]; then
    log "FATAL: expected checkpoint $TRAIN_CKPT not found after training. Stopping."
    set_status "FAILED"
    exit 1
fi
log "Checkpoint verified: $TRAIN_CKPT"

# ---------------------------------------------------------------------------
# Step 2: retrain the SNN attacker (privacy evaluation, VAL only).
# ---------------------------------------------------------------------------
log "STEP 2/3: retraining SNN ($SNN_CFG)"
if "$PY" retrain_SNN_v2.py \
        --config_path "$CFG_DIR" --config "$SNN_CFG" \
        > "$LOGDIR/retrain_snn.log" 2>&1; then
    log "STEP 2 finished OK."
else
    CODE=$?
    log "STEP 2 FAILED (exit $CODE). See $LOGDIR/retrain_snn.log -- stopping queue."
    set_status "FAILED"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: evaluate classifier utility on anonymized images (VAL fold).
# ---------------------------------------------------------------------------
log "STEP 3/3: evaluating classifier ($EVAL_CFG)"
if "$PY" eval_classifier_v2.py \
        --config_path "$CFG_DIR" --config "$EVAL_CFG" \
        > "$LOGDIR/eval_classifier.log" 2>&1; then
    log "STEP 3 finished OK."
else
    CODE=$?
    log "STEP 3 FAILED (exit $CODE). See $LOGDIR/eval_classifier.log -- stopping queue."
    set_status "FAILED"
    exit 1
fi

set_status "COMPLETE"
log "=== V2 PIPELINE COMPLETE. Results: privacy VAL AUC in archive/v2_snn_retrain_run1/, utility in archive/v2_eval_classifier_run1/summary.json ==="

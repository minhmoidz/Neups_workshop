#!/usr/bin/env bash
# ============================================================================
# Batch-ablation queue: waits for the V2 run1 pipeline to COMPLETE, then runs
# two control arms through the identical V2 stack (plain U-Net, feature loss
# OFF), differing ONLY in effective batch size:
#   Arm A: v2ctrl_b16acc1  -> batch 16, accumulation 1   (= B_dev conditions)
#   Arm B: v2ctrl_b16acc2  -> batch 16, accumulation 2   (= effective batch 32)
#
# Interpretation guide (privacy VAL AUC, lower = better anonymization):
#   A ~ 0.61 and B ~ 0.81 -> effective batch size is the main driver of the
#                            gap vs the paper reproduction.
#   A ~ B ~ 0.81          -> evaluation geometry is the driver; batch innocent.
#   Anything else         -> interaction effects; analyze per-arm losses.
#
# Completion signal: machine-readable status file written by the run1 queue
# (RUNNING / COMPLETE / FAILED). Grep-based detection of append-only logs is
# unsafe across queue restarts because historical failure lines would be
# misread as current state.
# ============================================================================

set -u

WORKDIR="/home/minhtt/Neups_workshop"
PY="$WORKDIR/.venv/bin/python"
POLL_SECONDS=300
RUN1_STATUS="$WORKDIR/research_runs/V2_QUEUE/status.txt"

LOGDIR="$WORKDIR/research_runs/BATCH_ABLATION_QUEUE"
mkdir -p "$LOGDIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/queue.log"
}

log "=== BATCH-ABLATION QUEUE STARTED (waiting for run1 completion) ==="

# Wait for the run1 pipeline to fully complete.
while true; do
    if [ ! -f "$RUN1_STATUS" ]; then
        log "run1 status file missing (queue not started yet); waiting ${POLL_SECONDS}s..."
        sleep "$POLL_SECONDS"
        continue
    fi
    STATUS=$(tr -d '[:space:]' < "$RUN1_STATUS" 2>/dev/null)
    case "$STATUS" in
        RUNNING|"")
            log "run1 RUNNING; waiting ${POLL_SECONDS}s..."
            sleep "$POLL_SECONDS"
            ;;
        FAILED)
            log "run1 reported FAILURE -- aborting ablation queue."
            exit 1
            ;;
        COMPLETE)
            log "run1 pipeline COMPLETE detected."
            break
            ;;
        *)
            log "unknown status '$STATUS'; treating as RUNNING, waiting..."
            sleep "$POLL_SECONDS"
            ;;
    esac
done

cd "$WORKDIR" || { log "FATAL: cannot cd"; exit 1; }

run_arm() {
    local ARM="$1"
    local TRAIN_CFG="config_files/config_anonymization_${ARM}.json"
    local SNN_CFG="config_files/config_retrainSNN_${ARM}.json"
    local EVAL_CFG="config_files/config_eval_classifier_${ARM}.json"
    local CKPT="./archive/${ARM}/generator_lowest_total_loss.pth"
    local EXP_DIR="./archive/${ARM}"

    # Same fail-closed os.mkdir in train_architecture_v2.py as run1: a
    # leftover dir from a previous killed/crashed attempt makes this arm
    # crash instantly on retry. Archive it instead of dying.
    if [ -d "$EXP_DIR" ]; then
        local STAMP
        STAMP=$(date '+%Y%m%d_%H%M%S')
        log "[$ARM] Found pre-existing $EXP_DIR from a previous attempt; moving it to ${EXP_DIR}_preempted_${STAMP} before restarting."
        mv "$EXP_DIR" "${EXP_DIR}_preempted_${STAMP}"
    fi

    log "[$ARM] STEP 1/3: training ($TRAIN_CFG)"
    if "$PY" train_architecture_v2.py --config_path "./config_files/" \
            --config "$(basename "$TRAIN_CFG")" \
            > "$LOGDIR/${ARM}_train.log" 2>&1; then
        log "[$ARM] STEP 1 OK."
    else
        log "[$ARM] STEP 1 FAILED -- aborting remaining arms."
        return 1
    fi

    if [ ! -f "$CKPT" ]; then
        log "[$ARM] FATAL: checkpoint $CKPT missing after training."
        return 1
    fi

    log "[$ARM] STEP 2/3: retraining SNN ($SNN_CFG)"
    if "$PY" retrain_SNN_v2.py --config_path "./config_files/" \
            --config "$(basename "$SNN_CFG")" \
            > "$LOGDIR/${ARM}_snn.log" 2>&1; then
        log "[$ARM] STEP 2 OK."
    else
        log "[$ARM] STEP 2 FAILED -- aborting remaining arms."
        return 1
    fi

    log "[$ARM] STEP 3/3: evaluating classifier ($EVAL_CFG)"
    if "$PY" eval_classifier_v2.py --config_path "./config_files/" \
            --config "$(basename "$EVAL_CFG")" \
            > "$LOGDIR/${ARM}_eval.log" 2>&1; then
        log "[$ARM] STEP 3 OK."
    else
        log "[$ARM] STEP 3 FAILED."
        return 1
    fi
    return 0
}

if run_arm "v2ctrl_b16acc1"; then
    log "Arm A (effective batch 16) complete."
else
    log "Arm A failed -- skipping Arm B to avoid wasted GPU-hours."
    exit 1
fi

if run_arm "v2ctrl_b16acc2"; then
    log "Arm B (effective batch 32) complete."
else
    log "Arm B failed."
    exit 1
fi

log "=== BATCH ABLATION COMPLETE. Compare archive/v2ctrl_b16acc1_snn/ vs archive/v2ctrl_b16acc2_snn/ privacy VAL AUCs. ==="

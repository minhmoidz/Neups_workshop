#!/bin/bash
cd /workspace

echo "== Stage 3 launch script started $(date) =="
echo "SMOKE_START" > /workspace/stage3_status.txt

python train_architecture.py --config_path /workspace/config_files/ --config config_smoke_entropy.json > /workspace/smoke_entropy.log 2>&1
SMOKE_EXIT=$?

if ! grep -q "Finished Training!" /workspace/smoke_entropy.log; then
  echo "SMOKE_FAIL exit=$SMOKE_EXIT" >> /workspace/stage3_status.txt
  tail -30 /workspace/smoke_entropy.log >> /workspace/stage3_status.txt
  echo "== Stage 3 ABORTED $(date) ==" >> /workspace/stage3_status.txt
  exit 1
fi

echo "SMOKE_OK exit=$SMOKE_EXIT" >> /workspace/stage3_status.txt

rm -rf /workspace/archive/smoke_test_entropy
rm -f /workspace/config_files/config_smoke_entropy.json

echo "TRAIN_START $(date)" >> /workspace/stage3_status.txt
python train_architecture.py --config_path /workspace/config_files/ --config config_anonymization_run3.json > /workspace/eval_retrain_run3.log 2>&1
TRAIN_EXIT=$?
echo "TRAIN_DONE exit=$TRAIN_EXIT $(date)" >> /workspace/stage3_status.txt

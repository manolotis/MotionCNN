python /home/manolotis/sandbox/scenario_based_evaluation/motionCNN/predict.py \
  --config /home/manolotis/sandbox/scenario_based_evaluation/motionCNN/code/configs/predict.yaml \
  --test-data-path "/home/manolotis/sandbox/scenario_based_evaluation/motionCNN/data/prerendered/testing" \
  --batch-size 8 \
  --n-jobs 2 \
  --n-shards 1 \
  --out-path "/home/manolotis/sandbox/scenario_based_evaluation/motionCNN/predictions/"\
  --model-name "xception71"
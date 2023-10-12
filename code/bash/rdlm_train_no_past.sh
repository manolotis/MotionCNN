MODEL_NAME=xception71
python /home/manolotis/sandbox/robustness_benchmark/motionCNN/train.py \
    --train-data "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/training/" \
    --dev-data "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/validation/" \
    --train-data-noisy "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/training_no_past/" \
    --dev-data-noisy "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/validation_no_past/" \
    --save /home/manolotis/sandbox/robustness_benchmark/motionCNN/trained_models/${MODEL_NAME}_no_past \
    --model ${MODEL_NAME} \
    --img-res 224 \
    --in-channels 25 \
    --time-limit 80 \
    --n-traj 6 \
    --lr 0.001 \
    --batch-size 128 \
    --n-epochs 1000 \
    --n-jobs 16 \
    --n-shards 16
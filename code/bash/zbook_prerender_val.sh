CUDA_VISIBLE_DEVICES=-1 python /home/manolotis/sandbox/scenario_based_evaluation/motionCNN/prerender.py \
    --data "/home/manolotis/sandbox/datasets/waymo_v1.1/uncompressed/tf_example/validation/" \
    --out "/home/manolotis/sandbox/scenario_based_evaluation/motionCNN/data/prerendered/validation/" \
    --use-vectorize \
    --n-shards 1 \
    --n-jobs 8

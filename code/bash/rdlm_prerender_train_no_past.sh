CUDA_VISIBLE_DEVICES=-1 python /home/manolotis/sandbox/robustness_benchmark/motionCNN/prerender.py \
    --data "/media/disk1/datasets/waymo/motion v1.0/uncompressed/tf_example/training/" \
    --out "/home/manolotis/datasets/waymo/motion v1.0/prerender/motionCNN/training_no_past/" \
    --n-shards 1 \
    --hide-target-past True \
    --n-jobs 50

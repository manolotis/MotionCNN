# Waymo challenge 2021: motion prediction

[Motion Prediction](https://waymo.com/open/challenges/2021/motion-prediction/)

## Dataset

Download
[datasets](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_0_0)
`uncompressed/tf_example/{training,validation,testing}`

## Prerender

Change paths to input dataset and output folders

```bash
python prerender.py
```

## Training

```bash
pythin train.py
```

## Submit

Follow [`submission.ipynb`](./submission.ipynb).

## Visualize predictions

Follow [`check_cnn.ipynb`](./check_cnn.ipynb).

## Useful links

* [kaggle lyft 3rd place solution](https://gdude.de/blog/2021-02-05/Kaggle-Lyft-solution)

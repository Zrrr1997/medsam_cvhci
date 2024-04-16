# CVHCI Submission to MedSAM on a Laptop Challenge


## Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone -b LiteMedSAM https://github.com/bowang-lab/MedSAM/`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`
5. Change to a new directory to clone our code
6. `git clone https://github.com/Zrrr1997/medsam_cvhci`



```bash
python CVPR24_MyModel_infer.py -i test_demo/imgs/ -o test_demo/segs
```


### Build Docker (TODO Change Models)

```bash
docker build -f Dockerfile -t litemedsam .
```

> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name litemedsam --rm -v $PWD/test_demo/imgs/:/workspace/inputs/ -v $PWD/test_demo/litemedsam-seg/:/workspace/outputs/ litemedsam:latest /bin/bash -c "sh predict.sh"
```

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save docker 

```bash
docker save litemedsam | gzip -c > cvhci.tar.gz
```

### Compute Metrics

```bash
python evaluation/compute_metrics.py -s test_demo/litemedsam-seg -g test_demo/gts -csv_dir ./metrics.csv
```


## Model Training

### Data preprocessing
1. Pre-process all `npz` files to `npy` files using the `npz_to_npy.py` script
```bash
python npz_to_npy.py \
    -npz_path data/FLARE22Train/ \ ## path to training images
    -npy_path data/npy \ ## path to training labels
```
2. Train using your model
```bash
python train_my_model.py \
    -device cuda:0 \
    -model mobileunet \
    -num_epochs 500 \
    -batch_size 1 \
    -data_root ./data/npy \
```




## Acknowledgements
We thank the authors of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT) for making their source code publicly available.


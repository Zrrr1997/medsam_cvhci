# CVHCI Submission to MedSAM on a Laptop Challenge

## Challenge
Official implementation for the MedSAM on a Laptop Challenge [Track 1: Bounding Box](https://www.codabench.org/competitions/1847/) and [Track 2: Scribbles](https://www.codabench.org/competitions/2566/) for the team **cvhci**.

## Installation

The codebase is tested with: `Ubuntu 22.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.2.1`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone -b LiteMedSAM https://github.com/bowang-lab/MedSAM/`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`
5. Change to a new directory to clone our code
6. `git clone https://github.com/Zrrr1997/medsam_cvhci`


### Bounding box inference
```bash
python CVPR24_MyModel_infer_docker.py -i test_demo/imgs/ -o test_demo/segs
```

### Scribble inference (make sure images have the 'scribble' keyword)
```bash
python CVPR24_LiteMedSAM_infer_scribble_docker.py -i test_demo_scribbles/imgs/ -o test_demo_scribbles/segs
```


### Build Dockers

```bash
docker build -f Dockerfile -t cvhci .
docker build -f Dockerfile -t cvhci_scribble .
```

> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name cvhci --rm -v $PWD/test_demo/imgs/:/workspace/inputs/ -v $PWD/test_demo/litemedsam-seg/:/workspace/outputs/ cvhci:latest /bin/bash -c "sh predict.sh"

docker container run -m 8G --name cvhci_scribble --rm -v $PWD/../../../Datasets/test_demo_scribbles/:/workspace/inputs/ -v $PWD/test_demo/litemedsam-seg/:/workspace/outputs/ cvhci_scribble:latest /bin/bash -c "sh predict_scribble.sh"
```

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save dockers 

```bash
docker save cvhci | gzip -c > cvhci.tar.gz

docker save cvhci_scribble | gzip -c -> cvhci_scrible.tar.gz
```

### Compute Metrics

```bash
python evaluation/compute_metrics.py -s test_demo/litemedsam-seg -g test_demo/gts -csv_dir ./metrics.csv
```


## Model Training (Only for MobileUNet Bounding Box Models)

### Data preprocessing
1. Pre-process all `npz` files to `npy` files using the `npz_to_npy.py` script
```bash
python npz_to_npy.py \
    -npz_path data/npz_files/ \ ## path to training images
    -npy_path data/npy \ ## path to training labels
```
2. Train a MobileUNet
```bash
python train_my_model.py -data_root ./data/npy -work_dir workdir_mobileunet/ -num_epochs 500 -batch_size 4 --crop_instances
```




## Acknowledgements
We thank the authors of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT), and [MobileUNet](https://github.com/lafith/Mobile-UNet)  for making their source code publicly available.

## Reference
```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={1--9},
  year={2024}
}
```
```
@article{jing2022mobile,
  title={Mobile-Unet: An efficient convolutional neural network for fabric defect detection},
  author={Jing, Junfeng and Wang, Zhen and R{\"a}tsch, Matthias and Zhang, Huanhuan},
  journal={Textile Research Journal},
  volume={92},
  number={1-2},
  pages={30--42},
  year={2022},
  publisher={SAGE Publications Sage UK: London, England}
}
```



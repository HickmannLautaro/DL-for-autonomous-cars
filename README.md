# sim2real: top-view
## Dual style transfer and autonomous driving agent discrete classifier proof of concept.
Based on the [Contrastive Unpaired Translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation/tree/1b25f54a2e098cb330e6ee7e4c13510fed44d027) FastCUT model and using the [OpenAI](https://github.com/openai/gym) simulation environment.

To pre-train the edge detection network use:

```python
 styletransfer/cut/train.py -dataroot ./datasets/road_tree_new_train --name new_trees/edge_detection_MSE --batch_size 32 --dataset_mode "conditional" --model "edge"  --n_epochs 150 --display_freq 100 --output_nc 1 --ngf 16  --edge_loss "MSE"
```
The edge detection model has to be present in the cut/checkpoint/experiment_name folder before training the style transfer model.
To train the dual style transfer model use:
```python
 styletransfer/cut/train.py --dataroot ./datasets/road_tree_new_train --name new_trees/styletransfer_MSE_histo_10_edges_10 --CUT_mode FastCUT --batch_size 4 --dataset_mode "conditional" --model "conditional_cut"  --netG "conditional_resnet_9" --netD "conditional" --display_freq 100 --lambda_hist 10 --lambda_edge 10 --edge_loss "MSE" --n_epochs 50
```

To run the simulation use 
```shell
simulation/play_evaluation.py
```

# Demo
## Complete driving (autonomous and per hand with different style transfers)
https://user-images.githubusercontent.com/50120027/125075755-51c25e00-e0bf-11eb-909b-6ca3f73de86c.mp4


## Style transfer comparison
![ezgif com-gif-maker](https://user-images.githubusercontent.com/50120027/125068879-49195a00-e0b6-11eb-9bde-81058f9dc6b0.gif)

## GradCAM and AD on different styles
https://user-images.githubusercontent.com/50120027/126092797-0a77215f-a102-4581-8f25-225daf198880.mp4

# Installation

On Ubuntu 16.04 and 18.04: 
```shell
apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
```

Create conda environment with

```shell
conda env create [--name envname] -f environment.yml
conda activate [envname or DL_CAR] 
```
[ ...] denotes optional

then install gym 
```shell
pip install gym
```

pyglet

```shell
pip install --upgrade pyglet
```

Install pytorch and torchvision 

dominate
```shell
pip install dominate
```
visdom 
```shell
pip install visdom
```

openCV
```shell
pip install opencv-python
```

scikit-image
```shell
conda install scikit-image
```

Optional for visualization:

plotly with jupyter lab support:
```shell
conda install -c plotly plotly
conda install jupyterlab "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly@4.14.3
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3

```

```shell
pip install pykeops
pip install geomloss
pip install kornia
conda install -c conda-forge ipympl
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
```

FID comparrison
```shell
pip install pytorch-fid
```

GradCAM [Class Activation Map methods implemented in Pytorch](https://github.com/jacobgil/pytorch-grad-cam)

```shell
pip install grad-cam
```
SSIM loss [https://github.com/Po-Hsun-Su/pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) is included.

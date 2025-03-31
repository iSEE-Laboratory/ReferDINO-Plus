# **ReferDINO-Plus**

ReferDINO-Plus: 2nd Solution for 4th PVUW MeViS Challenge at CVPR 2025

## Installation

------

Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

The input of SAM2 is based on the results of ReferDINO [[Annotations_intermediate](https://drive.google.com/drive/folders/15pf3_-zkDZlfks3tyv0eiiCe7V-QOhe4?usp=sharing)] , which we have uploaded to Google Drive.

The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
cd sam2
pip install -e .
pip install -r requirements.txt
```

## Getting Started

------

Executing the code is straightforward:

```bash
python refine_from_refdino.py --gids 0 1 2 3 4 5 6 7
```

The parameter `gids` should be set based on the number of GPUs available on your device. The code will automatically create a `refine_output` directory and save the results under it.


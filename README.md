# **ReferDINO-Plus: 2nd Solution for 4th PVUW MeViS Challenge at CVPR 2025**

* 📃[Technical Report](https://arxiv.org/pdf/2503.23509)

![](https://github.com/iSEE-Laboratory/ReferDINO-Plus/blob/main/visual.png)

## Model Overview
![](https://github.com/iSEE-Laboratory/ReferDINO-Plus/blob/main/model.png)

In this repository, we present the code of *SAM2-based Mask Enhancement* and *Conditional Mask Fusion*, which corresponds to the 2nd and 3rd stages of our solution.

Meanwhile, we provide our intermediate results from the 1st stage on [Google Drive](https://drive.google.com/drive/folders/15pf3_-zkDZlfks3tyv0eiiCe7V-QOhe4), which can be directly used to execute this code.

The code of ReferDINO will be available at [here](https://github.com/iSEE-Laboratory/ReferDINO) once it is ready, thanks for your patience 🫡.


## Installation

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


The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
cd sam2
pip install -e .
pip install -r requirements.txt
```

## Getting Started

Executing the code is straightforward:

```bash
python refine_from_refdino.py --gids 0 1 2 3 4 5 6 7
```

The parameter `gids` should be set based on the number of GPUs available on your device. The code will automatically create a `refine_output` directory and save the results under it.

## Acknowledgements

Our code is based on [SAM2](https://github.com/facebookresearch/sam2). It is a fantastic work.


## Citation
If you find our work helpful for your research, please consider citing our report and paper.
```bibtex
@inproceedings{liang2025referdino,
    title={ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations},
    author={Liang, Tianming and Lin, Kun-Yu and Tan, Chaolei and Zhang, Jianguo and Zheng, Wei-Shi and Hu, Jian-Fang},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2025}
}

@article{liang2025referdinoplus,
  title={ReferDINO-Plus: 2nd Solution for 4th PVUW MeViS Challenge at CVPR 2025},
  author={Liang, Tianming and Jiang, Haichao and Zheng, Wei-Shi and Hu, Jian-Fang},
  journal={arXiv preprint arXiv:2503.23509},
  year={2025}
}

```

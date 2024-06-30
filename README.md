
# Towards Collaborative Semantic Visual Navigation via Large Language Models

This is our ongoing work. We proposed a new framework to explore and search for the target in unknown environment based on Large Language Model. Our work is based on [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation) and [L3MVN](https://sites.google.com/view/l3mvn), implemented in PyTorch.

**Author:** Bangguo Yu, Kailai Li, Hamidreza Kasaei and Ming Cao

**Affiliation:** University of Groningen

## Abstract

Target-driven visual navigation in an unknown environment plays crucial roles towards reaching high-performance autonomy and human-machine interactions for intelligent robots. Most existing approaches for mapless visual navigation focus on single-robot operations, which  often lack  efficiency and robustness in complex environments. Meanwhile, policy learning for multi-robot collaboration is resource-demanding. To address these challenges, we propose Co-NavGPT, an innovative multi-robot framework integrating large language models (LLMs) as a global planner for collaborative visual navigation. We conduct experiments in synthetic environments for evaluation. Numerical results show superior performance of Co-NavGPT over existing approaches in terms of success rate and efficiency, not requiring the learning procedure and yet demonstrating great potential of exploiting LLMs in multi-robot collaboration. We open-source our current implementation at https://sites.google.com/view/co-navgpt.

![image-20200706200822807](img/framework.png)

## Installation

The code has been tested only with Python 3.10.8, CUDA 11.7.

1. Installing Dependencies
- We use adjusted versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below:

- Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt; 
python setup.py install --headless
python setup.py install # (for Mac OS)
```

- Installing habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/challenge-2022; 
pip install -e .
```

Back to the curent repo, and replace the habitat folder in habitat-lab rope for the multi-robot setting: 

```
mv -r multi-robot-setting/habitat enter-your-path/habitat-lab
```

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on torch v2.0.1, torchvision 0.15.2. 

- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration. If you are using conda:

2. Download HM3D_v0.2 datasets:

#### Habitat Matterport
Download [HM3D](https://aihabitat.org/datasets/hm3d/) dataset using download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d):
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

3. Download additional datasets

Download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) in RedNet/model path.


## Setup
Clone the repository and install other requirements:
```
git clone https://github.com/ybgdgh/Co-NavGPT
cd Co-NavGPT/
pip install -r requirements.txt
```

### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
Co-NavGPT/
  data/
    scene_datasets/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data
    objectgoal_hm3d/
        train/
        val/
        val_mini/
```


### For evaluation: 
For evaluating the framework, you need to setup your openai api keys in the 'exp_main_gpt.py', then run:
```
python exp_main_gpt.py
```


## Demo Video

[video](https://sites.google.com/view/co-navgpt)

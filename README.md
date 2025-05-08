# SLAM
SLAM: Towards Efficient Multilingual Reasoning via Selective Language Alignment

<p align="center">
  <a href="http://arxiv.org/abs/2501.03681"> 📃 Paper</a> | 
  <a href="https://huggingface.co/fmm170"> 🤗 Huggingface</a> 
</p>

### 🌚 Overview 
* This repository shows our latest work on the efficient expansion of model multilingual capabilities.

### 🛠️ Installation
To install this repository, follow these steps:
```
git clone git@github.com:fmm170/SLAM.git
cd SLAM/LLaMA-Factory
pip install -r requirements.txt
cd transformers
pip install -e .
```


### 🍀 Training
* finetuning MetaMath-7B / MetaMath-13B
```bash
bash MetaMath-7b.sh
bash MetaMath-13b.sh
```


### 🔍 Evaluation

We use the evaluation code provided by [Chen et al.](https://github.com/microsoft/MathOctopus)

* evaluating with mGSM
```bash
cd evaluate

bash mgsm.sh
```
* evaluating with mSVAMP
```bash
cd evaluate

bash msvamp.sh
```

* evaluating with xCSQA
```bash
cd evaluate

bash evaluate_xcsqa2.sh
```

## 🐛 Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email [yuchunfan_neu@outlook.com](yuchunfan_neu@outlook.com)

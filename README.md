##### Table of Content

1. [Introduction](#SMC-Smart-Cell)
1. [Datasets](#datasets)
1. [Getting Started](#getting-started)
	- [Requirements](#requirements)
	- [Usage Example](#usage)
1. [Training & Evaluation](#training-and-evaluation)
1. [Weakpoint](#weak-point)
1. [Improve ideas](#improve-ideas)
1. [Result](#result)

# SMC: Smart Cell 

- SMC is project to manage OP in factory, manager can monitor, measure OP's time study to balance line & prevent skip operation.
- In this project, I divided OP's action into 3 steps.

| ![steps.PNG](https://github.com/Quang111198/Prevent-OP-mising-operation/blob/7a47f832b87a6485a888dbe27d081140fa761f22/img/steps.PNG)|
|:--:|
| *SMC OP steps in one station.*|

First, I test model with availabel dataset. Details of the dataset construction, model architecture, and experimental results can be found [following link colab](https://colab.research.google.com/drive/1RtTYonaJ7ASX_ZMzcV3t_0jNktheKQF9?usp=sharing).

---

### Datasets

I introduce ✨ 2 datasets: **UCF50**, **UCF101**. Besides, this is my dataset [OP's step Dataset](https://www.dropbox.com/s/u3n76duuzbw537p/SMC_Project.rar?dl=0).

I will keep my dataset on Dropbox until out of space. Hope it can help u!!! 

---

### Getting Started

##### Requirements

- python
- tensorflow
- os-sys
- sklearn
- opencv-python

##### Installation

``` sh
# clone the repo
git clone https://github.com/Quang111198/Prevent-OP-mising-operation.git
cd Prevent-OP-mising-operation
```

##### Usage

➡️ *You can now try test UCF50 in Google Colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RtTYonaJ7ASX_ZMzcV3t_0jNktheKQF9?usp=sharing)*

---

### Training and Evaluation
Flowchart:

1. Data preparation: resized height & width each video frame, specify the number of frames of a video .

2. Training using LRCN model [LRCN paper](https://arxiv.org/abs/1411.4389?source=post_page---------------------------). Besides, you can test [ConvLSTM](https://arxiv.org/abs/1506.04214v1) or [Bi-LSTM](https://paperswithcode.com/method/bilstm). 
I tested and see LRCN is the most suitable with my data.

---

### Weak point:

1. With actions not label, machine still classify in 3 classes. 

2. Must improve algorithm if have noise like: other person or hand appear in action areas.

3. What happen if stuck in line.

---

### Improve ideas:

1. Tracking OP or hand (sort, deepsort, byte-track).

2. Add label "play" in break time. 

---

### Result 

Can't run here, pls download to see result!!! [here](https://github.com/Quang111198/Prevent-OP-mising-operation/tree/main/vid)


# Quantum Graph-based Object Detection (QGOD)
This is the description for QGOD.
This repository covers a Quantum Graph-based Object Detection framework for real-time road analysis in intelligent Cooperative Adaptive Cruise Control (CACC) systems, based on quantum machine learning and graph convolutional reasoning.

QGOD integrates the lightweight feature extraction capability of Quantum Convolutional Neural Networks (QCNNs) with the relational learning ability of Graph Convolutional Networks (GCNs). By encoding local quantum features through QCNNs and modeling inter-patch correlations via GCNs, QGOD achieves a compact feature representation while preserving reliable detection performance in resource-constrained vehicular environments. In addition, a Lyapunov-based adaptive controller dynamically adjusts the depth of the Parameterized Quantum Circuit (PQC) to balance inference accuracy and queue stability under real-time operating conditions.

## Demo Video
<!-- TODO: Replace with QGOD demo video link -->
![qgod_demo](https://github.com/user-attachments/assets/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)


## 0. References
This code is implemented based on the Quantum Convolution for Multi-Channel Object Detection (QCOD) framework, which itself extends "Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds" [Link](https://github.com/maudzung/SFA3D). QGOD extends QCOD by introducing a graph convolutional layer for relational reasoning across quantum-encoded image patches, together with a Lyapunov-based PQC depth controller.

To cite the baseline model, please refer to the following:
```
@misc{Super-Fast-Accurate-3D-Object-Detection-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{Super-Fast-Accurate-3D-Object-Detection-PyTorch}},
  howpublished = {\url{https://github.com/maudzung/SFA3D}},
  year =         {2020}
}
```

## 1. Hierarchy
We implement the proposed QGOD model, which combines Quantum Convolution, a Graph Convolutional layer, and a Lyapunov-based PQC depth controller for adaptive real-time detection.
The current repository structure is as follows.
```
${ROOT}
├── checkpoint/
│   ├── checkpoint_best_cnn.pth
│   ├── checkpoint_best_qcod_GCN_1007_class2-pqc2.pth
│   ├── checkpoint_best_qcod_GCN_1007_class2-pqc3.pth
│   ├── checkpoint_best_qcod_GCN_1011_class2-pqc1.pth
│   └── checkpoint_best_qcod_GCN_1011_class2-pqc5.pth
├── dataset/
│   └── dataset.ipynb
├── train/
│   ├── train_qcnn.ipynb
│   └── train_qgod.ipynb
├── .gitignore
└── README.md
```

> **Note:** The image dataset (`dataset/Highway-classification-1-downsampled/`) is excluded from this repository due to its size. See [Section 2](#2-dataset) for details.

## 2. DATASET
QGOD is trained and evaluated on a highway-scene dataset consisting of cars and motorcycles, collected for real-time CACC road analysis.

- Training samples: **1,478** images  
- Testing samples: **335** images  
- Classes: **car**, **motorcycle**  
- Resolution: **416 × 416** (low-resolution setting)
- Target object size: **ultra-small (10–50 pixels)**
- Preprocessing: all images are converted to **grayscale** prior to augmentation, to emphasize localization performance under resource-constrained vehicular conditions.

The class-agnostic **mAP@0.5** is used as the primary evaluation metric, focusing on localization rather than classification, which is appropriate given the ultra-small object sizes and low input resolution of the target scenario.

## 3. Training
**(Note!) Batch Size → Fixed to 1.** For fast calculation, the operations inside the Quantum Convolutional layer are batched internally across patches rather than across samples. You can modify this behavior by increasing the total patch dimension in `QCNN.py`.

### 3.1 Model Variants
Three models are provided for comparison, all sharing the same classical CNN backbone and detection head:

| Model | Quantum Conv. | Graph Conv. | Lyapunov PQC Control |
|-------|:---:|:---:|:---:|
| Classical CNN | ❌ | ❌ | ❌ |
| QCOD | ✅ | ❌ | ❌ |
| **QGOD (proposed)** | ✅ | ✅ | ✅ |

### 3.2 Quantum Convolutional Layer
The input image is divided into patches and encoded into quantum states. Quantum operations use rotation gates (Rx, Ry, Rz) to map real-valued inputs to quantum states, CNOT gates to capture local correlations via entanglement, and CU3 gates for data-dependent transformations. This design achieves high representational efficiency with significantly fewer parameters.

### 3.3 Graph Convolutional Layer (QGOD only)
The quantum feature map is transformed into a grid-based graph where each cell is a node connected to its four neighbors and itself. The GCN aggregates neighboring node features as

```
h_v^(l+1) = σ( Σ_{u ∈ N(v) ∪ {v}} Â_vu · W^(l) · h_u^(l) )
```

thereby expanding local quantum features into global contextual representations and enhancing spatial consistency across patches.

### 3.4 Lyapunov-based PQC Depth Control
To mitigate the inherent accuracy–latency trade-off of QCNNs, a discrete-time queue `Q[t]` models incoming image frames, and a drift-plus-penalty (DPP) policy selects the PQC depth `α[t]` at each time slot:

```
α*[t] = arg max { V · A(α[t]) − Q[t] · b(α[t]) }
```

- Larger `V` → prioritizes **accuracy** (deeper PQC, higher latency).
- Smaller `V` → prioritizes **queue stability** (shallower PQC, lower latency).

The controller ensures a bounded time-averaged queue backlog while driving the long-term average analysis accuracy toward near-optimality.

### 3.5 Training Commands
```bash
# Train classical CNN baseline
python sfa/train.py

# Train QCOD (quantum convolution only)
python sfa/train_QCOD.py

# Train QGOD (quantum + GCN + Lyapunov PQC control)
python sfa/train_QGOD.py
```

## 4. Experimental Results (Stage 1)

### 4.1 Accuracy vs. Efficiency (1 PQC layer)

| Model | mAP@0.5 (%) | # Parameters | Inference Time (sec) |
|-------|:---:|:---:|:---:|
| Classical CNN | 51.99 | 156,424 | 1.00 × 10⁻² |
| QCOD (32 channels) | 50.46 | 43,454 | 1.53 × 10⁻² |
| QCOD (64 channels) | 51.52 | 49,214 | 1.89 × 10⁻² |
| **QGOD (32 channels)** | **51.85** | **44,384** | **0.91 × 10⁻²** |

QGOD achieves comparable or higher accuracy than both the classical CNN and QCOD, while using roughly **3.5× fewer parameters** than the classical CNN and delivering the **shortest inference time** overall.

### 4.2 Effect of PQC Depth on QGOD

| PQC Layers | Accuracy (%) | Inference Time (sec) |
|:---:|:---:|:---:|
| 1 | 51.85 | 0.91 × 10⁻² |
| 3 | 54.12 | 1.70 × 10⁻² |
| 5 | 57.15 | 1.93 × 10⁻² |

Deeper PQCs improve accuracy but increase inference latency — exactly the trade-off that the Lyapunov controller resolves at runtime according to the current queue state.

## REFERENCE
[1] "Joint Optimization of Quantum-Inspired Real-Time Road Analysis and AoI-Aware Data Transmission for Intelligent CACC Systems" (this work) <br/>
[2] "Object Detector for Autonomous Vehicles Based on Improved Faster RCNN": [Improved Faster R-CNN](https://github.com/Ziruiwang409/improved-faster-rcnn/blob/main/README.md) <br/>
[3] "Torch-quantum": [QNN Implementation](https://github.com/mit-han-lab/torchquantum) <br/>
[4] "Fast Quantum Convolutional Neural Networks for Low-Complexity Object Detection in Autonomous Driving Applications", *IEEE Transactions on Mobile Computing*, 2025. <br/>
[5] "Dynamic Multi-PQC Quantum Convolutional Neural Network for Real-Time Pothole Detection", *IEEE SPAWC*, 2025. <br/>
[6] "Quantum-Inspired Multi-Scale Object Detection in UAV Imagery", *IEEE Access*, 2025. <br/>
[7] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet) <br>
[8] RTM3D: [PyTorch Implementation](https://github.com/maudzung/RTM3D) <br>
[9] Libra_R-CNN: [PyTorch Implementation](https://github.com/OceanPang/Libra_R-CNN) <br>

*Lyapunov optimization for real-time ML systems:* <br>
[10] M. J. Neely, "Stochastic Network Optimization with Application to Communication and Queueing Systems," Morgan & Claypool, 2010.

*Graph Convolutional Networks:* <br>
[11] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," *ICLR*, 2017.

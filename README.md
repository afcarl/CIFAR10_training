This repository contains Torch code for training VGG models on CIFAR-10 dataset. This code was written as a part of Image Understanding course in Universiy of Maryland. Two training strategies were explored - use of auxiliary classifiers and extreme augmentation and both these methods showed improved performance. To illustrate the power of CNNs, we also implemented Rotation Invariant Local Binary Pattern features and trained an SVM on these features.

Summary of the results:
VGG - 92.5%
VGG+auxiliary classifiers - 93.05%
VGG+extreme augmentation  - 93.68%

LBP+SVM - 35.63%
Region based LBP+SVM - 38.37 %


# Object recognition on CIFAR-10 dataset

This repository contains Torch code for training VGG models on CIFAR-10 dataset. This code was written as a part of Image Understanding course at Universiy of Maryland, College Park. Two training strategies were explored - use of auxiliary classifiers and extreme augmentation and both these methods showed improved performance. To illustrate the power of CNNs, we trained a Support Vector Machine on LBP features and compared their performance to CNNs. We found that the CNNs achieve more than twice the accuracy than these hand-crafted feature extraction methods.

## Summary of the results:
1. VGG - 92.5%
2. VGG+auxiliary classifiers - 93.05%
3. VGG+extreme augmentation  - 93.68%
4. LBP+SVM - 35.63%
5. Region based LBP+SVM - 38.37 %


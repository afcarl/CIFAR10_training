%% Code for training Support Vector Machine on LBP features
%  Author: Yogesh Balaji
% 
%  Input: Feature vectors computed (LBP in this case)

clear all;
clc;
addpath('/scratch0/Softwares/libsvm/matlab/');
load '/scratch0/dataset/cifar/cifar_LBP_feat_uniform_region.mat';

model = svmtrain(Training_feat{2}, Training_feat{1}, '-t 2 -g 0.5 -c 8');
[predicted_label, accuracy, decision] = svmpredict(Test_feat{2}, Test_feat{1}, model);

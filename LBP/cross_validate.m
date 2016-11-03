%% Code for training Support Vector Machine on LBP features
%  Author: Yogesh Balaji
%  Code for cross validation of SVM
%  Input: Feature vectors computed (LBP in this case)

% We cross validate parameters C and gamma for RBF kernel

clear all;
clc;
addpath('/scratch0/Softwares/libsvm/matlab/');
%load '/scratch0/dataset/cifar/cifar_LBP_feat.mat';
load '/scratch0/dataset/cifar/cifar_LBP_feat_uniform_region.mat';

trsize = size(Training_feat{2},1);
index = randperm(trsize);

trainSize = ceil(trsize*0.85);
valSize = trsize - trainSize;

train_feat = Training_feat{1}(index(1:trainSize),:);
train_labels = Training_feat{2}(index(1:trainSize));

val_feat = Training_feat{1}(index(trainSize+1:end),:);
val_labels = Training_feat{2}(index(trainSize+1:end));

cvals = 2.^[-3:7];
gamma_vals = 2.^[-5:5];

c_array = [];
g_array = [];
accuracy_array = [];
fileID = fopen('validation_uniform.txt','w');
for i = 1:length(cvals)
	for j = 1:length(gamma_vals)
		i
		j
		c = cvals(i);
		g = gamma_vals(j);
		model = svmtrain(train_labels, train_feat, ['-t 2 -h 0 -g ' num2str(g) ' -c ' num2str(c)]);
		[predicted_label, accuracy, decision] = svmpredict(val_labels, val_feat, model);
		c_array = [c_array; c]
		g_array = [g_array; g]
		accuracy_array = [accuracy_array; accuracy(1)]
		fprintf(fileID, '%f %f %f \n', c, g, accuracy(1));
	end
end
fclose(fileID);
save('validate.mat', 'c_array', 'g_array', 'accuracy_array');

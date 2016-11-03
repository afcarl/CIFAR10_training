%% Code for extracting LBP features from CIFAR dataset

clear all;
clc;

load '/scratch0/dataset/cifar/cifar_images.mat';
Save_path = '/scratch0/dataset/cifar/';

trsize = length(trainData{2});
tesize = length(testData{2});

features_train = zeros(trsize, 59*4);
features_test = zeros(tesize, 59*4);

for i=1:trsize
	i
	img = uint8(squeeze(trainData{1}(i,:,:,:)));
	feat = extract_LBP_features_regionwise(img);
    features_train(i,:) = feat';
end

for i=1:tesize
	i
	img = uint8(squeeze(testData{1}(i,:,:,:)));
	feat = extract_LBP_features_regionwise(img);
	features_test(i,:) = feat';
end

Training_feat = cell(2,1);
Test_feat = cell(2,1);
Training_feat{1} = features_train;
Training_feat{2} = trainData{2};
Test_feat{1} = features_test;
Test_feat{2} = testData{2};

save([Save_path 'cifar_LBP_feat_uniform_region.mat'], 'Training_feat', 'Test_feat');

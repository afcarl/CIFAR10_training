%% Code for extracting CIFAR images
%  Author: Yogesh Balaji

clear all;
clc;

% Specify the paths
Data_path = '/scratch0/dataset/cifar/cifar-10-batches-mat/';
Save_path = '/scratch0/dataset/cifar/cifar_torch_images/';

% Cell for storing the data 
trsize = 50000;
tesize = 10000;
trainData = {2};
testData = {2};

train_batches = {'data_batch_1.mat', 'data_batch_2.mat', 'data_batch_3.mat', 'data_batch_4.mat', 'data_batch_5.mat'};
test_batches = {'test_batch.mat'};


% extracting training data

dat = zeros(trsize,32,32,3);
lab = zeros(trsize,1);
counter = 1;

for bnum=1:length(train_batches)
	clear labels;
	clear data;
    load([Data_path train_batches{bnum}]);
    
    for index=1:10000
        
        imgdata = data(index,:);
        img = uint8(zeros(32,32,3));
        imgdata = reshape(imgdata,32,32*3)';
        img(:,:,1) = imgdata(1:32,:);
        img(:,:,2) = imgdata(33:32*2,:);
        img(:,:,3) = imgdata(65:32*3,:);
        
		imwrite(img, [Save_path 'train_images/' num2str(counter) '.png']);
        lab(counter,1) = labels(index);
        counter = counter+1;
    end;
end;
save([Save_path 'train_labels.mat'], 'lab');


% extracting test data

dat = zeros(tesize,32,32,3);
lab = zeros(tesize,1);
counter = 1;

for bnum=1:length(test_batches)
	clear labels;
	clear data;
    load([Data_path test_batches{bnum}]);
    
    for index=1:10000
        
        imgdata = data(index,:);
        img = uint8(zeros(32,32,3));
        imgdata = reshape(imgdata,32,32*3)';
        img(:,:,1) = imgdata(1:32,:);
        img(:,:,2) = imgdata(33:32*2,:);
        img(:,:,3) = imgdata(65:32*3,:);
        
        imwrite(img, [Save_path 'test_images/' num2str(counter) '.png']);
        lab(counter,1) = labels(index);
        counter = counter+1;
    end;
end;
save([Save_path 'test_labels.mat'], 'lab');


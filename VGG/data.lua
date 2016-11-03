require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator
matio = require 'matio'

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all'
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> downloading dataset')

classes = {}
for i=1,10 do
	classes[i] = i
end

local data = torch.load('/scratch0/dataset/cifar/cifar10_whitened.t7')

function hflip(x)
   return torch.random(0,1) == 1 and x or image.hflip(x)
end

function randomcrop(x)
   local pad = opt.randomcrop
   if opt.randomcrop_type == 'reflection' then
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float()
   elseif opt.randomcrop_type == 'zero' then
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   else
      error'unknown mode'
   end

   local imsize = opt.imageSize
   local padded = module:forward(x)
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)
   return padded:narrow(3,x,imsize):narrow(2,y,imsize)
end

function augment_data(x)
	x = x:float()
	for i=1,x:size(1) do
		x[i] = hflip(x[i])
		x[i] = randomcrop(x[i])
	end
	x = x:cuda()
	return x:clone()
end		

-- loading and preprocessing data

train_labels = matio.load('/scratch0/dataset/cifar/cifar_torch_images/train_labels.mat', 'lab')
test_labels = matio.load('/scratch0/dataset/cifar/cifar_torch_images/test_labels.mat', 'lab')

trsize = train_labels:size()[1]
tesize = test_labels:size()[1]

trainData = {
	data = torch.Tensor(trsize,3,32,32),
	labels = torch.Tensor(trsize),
	size = function() return trsize end
}
testData = {
	data = torch.Tensor(tesize,3,32,32),
	labels = torch.Tensor(tesize),
	size = function() return tesize end
}

local meanstd = {mean = {125.3/255, 123.0/255, 113.9/255}, std  = {63.0/255,  62.1/255,  66.7/255}}

-- Loading the images

for i = 1,trainData:size() do
	
	img = image.load('/scratch0/dataset/cifar/cifar_torch_images/train_images/' .. i .. '.png')
	for ch = 1,3 do
		img[ch]:add(-meanstd.mean[ch])
		img[ch]:div(meanstd.std[ch])
	end
	trainData.data[i] = img
end

for i = 1,testData:size() do
	
	img = image.load('/scratch0/dataset/cifar/cifar_torch_images/test_images/' .. i .. '.png')
	for ch = 1,3 do
		img[ch]:add(-meanstd.mean[ch])
		img[ch]:div(meanstd.std[ch])
	end
	testData.data[i] = img
end

trainData.labels = train_labels+1
testData.labels = test_labels+1

print(trainData.labels)
-- Exports
return {
   trainData = trainData,
   testData = testData,
   classes = classes
}


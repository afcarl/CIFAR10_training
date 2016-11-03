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

function zoomout(x)
	-- We shall crop 24X24 sized patch and zoom it out
	
	local xinit = torch.random(1,8)
	local yinit = torch.random(1,8)
	
	local cropped_img = x:narrow(3,xinit,24):narrow(2,yinit,24)
	local zoomed = image.scale(cropped_img, '32x32')
	return zoomed
end
	
function augment_data(x,y)
	x = x:float()
	local xnew = torch.Tensor(4*x:size(1),x:size(2),x:size(3),x:size(4))
	local ynew = torch.Tensor(4*y:size(1))
	for i=1,x:size(1) do
		xnew[4*i-3] = x[i]
		ynew[4*i-3] = y[i]
		
		xnew[4*i-2] = hflip(x[i])
		ynew[4*i-2] = y[i]
		
		xnew[4*i-1] = zoomout(x[i])
		ynew[4*i-1] = y[i]
		
		xnew[4*i] = randomcrop(x[i])
		ynew[4*i] = y[i]
	end
	
	xnew = xnew:cuda()
	ynew = ynew:cuda()
	return xnew,ynew
end		

-- loading and preprocessing data

train_labels = matio.load('/gleuclid/yogesh/cifar/cifar_torch_images/train_labels.mat', 'lab')
test_labels = matio.load('/gleuclid/yogesh/cifar/cifar_torch_images/test_labels.mat', 'lab')

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
	
	img = image.load('/gleuclid/yogesh/cifar/cifar_torch_images/train_images/' .. i .. '.png')
	for ch = 1,3 do
		img[ch]:add(-meanstd.mean[ch])
		img[ch]:div(meanstd.std[ch])
	end
	trainData.data[i] = img
end

for i = 1,testData:size() do
	
	img = image.load('/gleuclid/yogesh/cifar/cifar_torch_images/test_images/' .. i .. '.png')
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


require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'
require 'nngraph'

model1 = nn.Sequential()
model2 = nn.Sequential()
model3 = nn.Sequential()
model4 = nn.Sequential()

classifier1 = nn.Sequential()
classifier2 = nn.Sequential()
classifier3 = nn.Sequential()


model1:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1):noBias())
model1:add(cudnn.SpatialBatchNormalization(64,1e-3))
model1:add(cudnn.ReLU(true))
model1:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1):noBias())
model1:add(cudnn.SpatialBatchNormalization(64,1e-3))
model1:add(cudnn.ReLU(true))
model1:add(cudnn.SpatialMaxPooling(2,2,2,2))

model1:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1):noBias())
model1:add(cudnn.SpatialBatchNormalization(128,1e-3))
model1:add(cudnn.ReLU(true))
model1:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1):noBias())
model1:add(cudnn.SpatialBatchNormalization(128,1e-3))
model1:add(cudnn.ReLU(true))
model1:add(cudnn.SpatialMaxPooling(2,2,2,2))

classifier1 = nn.Sequential()
classifier1:add(cudnn.SpatialMaxPooling(8,8,1,1):ceil())
classifier1:add(nn.View(128))
classifier1:add(nn.Linear(128,128))
classifier1:add(nn.ReLU())
classifier1:add(nn.Dropout(0.5))
classifier1:add(nn.Linear(128,10))

model2:add(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1):noBias())
model2:add(cudnn.SpatialBatchNormalization(256,1e-3))
model2:add(cudnn.ReLU(true))
model2:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1):noBias())
model2:add(cudnn.SpatialBatchNormalization(256,1e-3))
model2:add(cudnn.ReLU(true))
model2:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1):noBias())
model2:add(cudnn.SpatialBatchNormalization(256,1e-3))
model2:add(cudnn.ReLU(true))
model2:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1):noBias())
model2:add(cudnn.SpatialBatchNormalization(256,1e-3))
model2:add(cudnn.ReLU(true))
model2:add(cudnn.SpatialMaxPooling(2,2,2,2))

classifier2 = nn.Sequential()
classifier2:add(cudnn.SpatialAveragePooling(4,4,1,1):ceil())
classifier2:add(nn.View(256))
classifier2:add(nn.Linear(256,256))
classifier2:add(nn.ReLU())
classifier2:add(nn.Dropout(0.5))
classifier2:add(nn.Linear(256,10))

model3:add(cudnn.SpatialConvolution(256,512,3,3,1,1,1,1):noBias())
model3:add(cudnn.SpatialBatchNormalization(512,1e-3))
model3:add(cudnn.ReLU(true))
model3:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model3:add(cudnn.SpatialBatchNormalization(512,1e-3))
model3:add(cudnn.ReLU(true))
model3:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model3:add(cudnn.SpatialBatchNormalization(512,1e-3))
model3:add(cudnn.ReLU(true))
model3:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model3:add(cudnn.SpatialBatchNormalization(512,1e-3))
model3:add(cudnn.ReLU(true))
model3:add(cudnn.SpatialMaxPooling(2,2,2,2))

classifier3 = nn.Sequential()
classifier3:add(cudnn.SpatialAveragePooling(2,2,1,1):ceil())
classifier3:add(nn.View(512))
classifier3:add(nn.Linear(512,512))
classifier3:add(nn.ReLU())
classifier3:add(nn.Dropout(0.5))
classifier3:add(nn.Linear(512,10))

model4:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model4:add(cudnn.SpatialBatchNormalization(512,1e-3))
model4:add(cudnn.ReLU(true))
model4:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model4:add(cudnn.SpatialBatchNormalization(512,1e-3))
model4:add(cudnn.ReLU(true))
model4:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model4:add(cudnn.SpatialBatchNormalization(512,1e-3))
model4:add(cudnn.ReLU(true))
model4:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model4:add(cudnn.SpatialBatchNormalization(512,1e-3))
model4:add(cudnn.ReLU(true))
model4:add(cudnn.SpatialAveragePooling(2,2,2,2):ceil())
model4:add(nn.View(512))
model4:add(nn.Linear(512,10))


local input = nn.Identity()()

local output11 = model1(input)
local output12 = classifier1(output11)

local output21 = model2(output11)
local output22 = classifier2(output21)

local output31 = model3(output21)
local output32 = classifier3(output31)

local output41 = model4(output31)

model = nn.gModule({input}, { output12, output22, output32, output41})

criterion = nn.ParallelCriterion()
criterion:add(nn.CrossEntropyCriterion())
criterion:add(nn.CrossEntropyCriterion())
criterion:add(nn.CrossEntropyCriterion())
criterion:add(nn.CrossEntropyCriterion())

model = model:cuda()
criterion = criterion:cuda()


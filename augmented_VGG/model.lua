require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'

model = nn.Sequential()

model:add(cudnn.SpatialConvolution(3,64,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(64,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64,64,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(64,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))

model:add(cudnn.SpatialConvolution(64,128,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(128,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(128,128,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(128,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))

model:add(cudnn.SpatialConvolution(128,256,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(256,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(256,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(256,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(256,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))

model:add(cudnn.SpatialConvolution(256,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))


model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1):noBias())
model:add(cudnn.SpatialBatchNormalization(512,1e-3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(2,2,2,2):ceil())

model:add(nn.View(512))
model:add(nn.Linear(512,10))


model = model:cuda()

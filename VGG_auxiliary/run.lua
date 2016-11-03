require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'
require 'cunn'
c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "/scratch0/Projects/CMSC828G/results/results_vgg_aux")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 0.1)        learning rate
   --learningRateDecay        (default 0)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 60)          epoch step
   --model                    (default model)     model name
   --max_epoch                (default 200)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
   --num_classes              (default 10)			  number of classes
   --hflip	  			      (default true)        horizontal flip
   --randomcrop		          (default 4)			 cropping factor
   --imageSize		          (default 32) 	     size of the image
   --randomcrop_type	      (default zero)		 crop type
]]

confusion1 = optim.ConfusionMatrix(10)
confusion2 = optim.ConfusionMatrix(10)
confusion3 = optim.ConfusionMatrix(10)
confusion4 = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


local data = dofile 'data.lua'
dofile 'model.lua'

parameters,gradParameters = model:getParameters()

dofile 'train.lua'
dofile 'test.lua'


for i=1,opt.max_epoch do
  train(data.trainData)
  test(data.testData)
end

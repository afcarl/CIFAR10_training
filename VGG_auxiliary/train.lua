local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

function train(trainData)
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/5 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = trainData.data:index(1,v)
    inputs = augment_data(inputs)
    targets:copy(trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, {targets, targets, targets, targets})
      local df_do = criterion:backward(outputs, {targets, targets, targets, targets})
      model:backward(inputs, df_do)
	  
      confusion1:batchAdd(outputs[1], targets)
	  confusion2:batchAdd(outputs[2], targets)
	  confusion3:batchAdd(outputs[3], targets)
	  confusion4:batchAdd(outputs[4], targets)
	  
      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion1:updateValids()
  confusion2:updateValids()
  confusion3:updateValids()
  confusion4:updateValids()
  
  print(('Train accuracy 1: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion1.totalValid * 100, torch.toc(tic)))
  print(('Train accuracy 2: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion2.totalValid * 100, torch.toc(tic)))
  print(('Train accuracy 3: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion3.totalValid * 100, torch.toc(tic)))
  print(('Train accuracy 4: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion4.totalValid * 100, torch.toc(tic)))

  train_acc = confusion4.totalValid * 100

  confusion1:zero()
  confusion2:zero()
  confusion3:zero()
  confusion4:zero()
  epoch = epoch + 1
end


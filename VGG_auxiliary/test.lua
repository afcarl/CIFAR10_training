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

function test(testData)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,testData.data:size(1),bs do
    local outputs = model:forward(testData.data:narrow(1,i,bs):cuda())
    confusion1:batchAdd(outputs[1], testData.labels:narrow(1,i,bs))
    confusion2:batchAdd(outputs[2], testData.labels:narrow(1,i,bs))
    confusion3:batchAdd(outputs[3], testData.labels:narrow(1,i,bs))
    confusion4:batchAdd(outputs[4], testData.labels:narrow(1,i,bs))
  end

  confusion1:updateValids()
  confusion2:updateValids()
  confusion3:updateValids()
  confusion4:updateValids()
  
  print('Test accuracy1:', confusion1.totalValid * 100)
  print('Test accuracy2:', confusion2.totalValid * 100)
  print('Test accuracy3:', confusion3.totalValid * 100)
  print('Test accuracy4:', confusion4.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion4.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion4)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 50 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model)
    --torch.save(filename, model:get(3):clearState())
  end

  confusion1:zero()
  confusion2:zero()
  confusion3:zero()
  confusion4:zero()
end


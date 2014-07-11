require 'torch'
require 'nn'

-- initial vector creation assuming an input vector with 10 values, 5 of which are word ids and 5 are caps ids

ltw = nn.LookupTable(130000, 50)
ltc = nn.LookupTable(5, 5)
pt = nn.ParallelTable()
pt:add(ltw)
pt:add(ltc)
jt = nn.JoinTable(2)
rs2 = nn.Reshape(275)

mlp = nn.Sequential()
mlp:add(pt)
mlp:add(jt)
mlp:add(rs2)

-- the NN layers
ll1 = nn.Linear(275, 300)
hth = nn.HardTanh()
ll2 = nn.Linear(300, 17)
lsm = nn.LogSoftMax()

mlp:add(ll1)
mlp:add(hth)
mlp:add(ll2)
mlp:add(lsm)

trainSize = 172389
testSize = 44462
-- create training data set
inputFile = torch.DiskFile('ner_training.txt', 'r')
inputLine = torch.IntStorage(10)

dataset = {}
function dataset:size() return trainSize end

for i=1,dataset:size() do 
   inputFile:readInt(inputLine)
   local input = torch.Tensor(10)
   for j=1,10 do 
      input[j] = inputLine[j]
   end

   local newInput = nn.SplitTable(1):forward(nn.Reshape(2,5):forward(input))
   local label = inputFile:readInt()

   dataset[i] = {newInput, label}

end

inputFile:close()

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)


-- Testing

outputFile = torch.DiskFile('ner_results.txt', 'w')

testFile = torch.DiskFile('ner_testing.txt', 'r')
inputLine = torch.IntStorage(10)

for i=1,testSize do 
   testFile:readInt(inputLine)
   local label = testFile:readInt()
   local input = torch.Tensor(10)
   for j=1,10 do 
      input[j] = inputLine[j]
   end
   local newInput = nn.SplitTable(1):forward(nn.Reshape(2,5):forward(input))
   output = mlp:forward(newInput)

   local outputLabel = 1;
   local outputValue = -1000;
   for k=1,17 do
      if output[k] > outputValue then
	 outputLabel = k;
	 outputValue = output[k];
      end
   end

   outputFile:writeInt(outputLabel);
      
end

testFile:close()
outputFile:close()
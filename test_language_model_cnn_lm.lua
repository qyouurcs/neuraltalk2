--[[
Unit tests for the LanguageModel implementation, making sure
that nothing crashes, that we can overfit a small dataset
and that everything gradient checks.
--]]

require 'torch'
require 'nn'

require 'misc.LanguageModelCNN_Only'
require 'misc.LanguageModelCriterion'
TDNN = require 'model.TDNN'
LSTMTD = require 'model.LSTMTD'
HighwayMLP = require 'model.HighwayMLP'

local gradcheck = require 'misc.gradcheck'

char_to_ix = {}
char_to_ix['{'] = 2
char_to_ix[' '] = 1
char_to_ix['}'] = 3
char_to_ix['#'] = 4
char_to_ix['t'] = 5
char_to_ix['r'] = 6
char_to_ix['a'] = 7
char_to_ix['s'] = 8


c_cnt = 9
for i = 65,65+26 do
    char_to_ix[string.char(i)] = c_cnt
    c_cnt = c_cnt + 1
end


ix_to_word = {['1']='trats',['2']='HELLO',['3']='WORLD',['4']='NIHAO', ['5'] = 'SHIJIE',['10']="OPT", ['6'] = "MAN", ['7'] = "AN", ['8'] = "THE", ['9'] = "GOOD"}


local tests = {}
local tester = torch.Tester()

-- validates the size and dimensions of a given 
-- tensor a to be size given in table sz
function tester:assertTensorSizeEq(a, sz)
  tester:asserteq(a:nDimension(), #sz)
  for i=1,#sz do
    tester:asserteq(a:size(i), sz[i])
  end
end

-- Test the API of the Language Model
local function forwardApiTestFactory(dtype)
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local function f()
    -- create LanguageModel instance
    opt = {}
    opt.vocab_size = 10

    opt.char_vocab_size = 31
    opt.max_word_l = 10
    opt.feature_maps = '{50,100,150,200,200,200,200}'
    opt.kernels =  '{1,2,3,4,5,6,7}'
    opt.char_vec_size = 15
    opt.use_words = 0
    opt.use_chars = 1
    opt.highway_layers = 2
    opt.hsm = 0
    opt.batch_norm = 0
    opt.gpuid = -1
  loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
  loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes



    opt.input_encoding_size = 11
    opt.rnn_size = 8
    opt.num_layers = 2
    opt.dropout = 0
    opt.seq_length = 7
    opt.batch_size = 10
    local lm = nn.LanguageModelCNN_Only(opt)
    local crit = nn.LanguageModelCriterion()
    lm:type(dtype)
    crit:type(dtype)
    
    -- construct some input to feed in
    local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
    seq[{ {4, 7}, 1 }] = 0
    seq[{ {5, 7}, 6 }] = 0

    local seq_chars = torch.LongTensor(opt.seq_length, opt.batch_size, opt.max_word_l):random(opt.char_vocab_size-2)
    -- make sure seq can be padded with zeroes and that things work ok
    local output = lm:forward{seq, seq_chars, char_to_ix}
    tester:assertlt(torch.max(output:view(-1)), 0) -- log probs should be <0

    -- the output should be of size (seq_length + 1, batch_size, vocab_size + 1)
    -- where the +1 is for the special END token appended at the end.
    tester:assertTensorSizeEq(output, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})

    local loss = crit:forward(output, seq)

    local gradOutput = crit:backward(output, seq)
    tester:assertTensorSizeEq(gradOutput, {opt.seq_length+2, opt.batch_size, opt.vocab_size+1})

    -- make sure the pattern of zero gradients is as expected
    local gradAbs = torch.max(torch.abs(gradOutput), 3):view(opt.seq_length+2, opt.batch_size)
    local gradZeroMask = torch.eq(gradAbs,0)
    local expectedGradZeroMask = torch.ByteTensor(opt.seq_length+2,opt.batch_size):zero()
    expectedGradZeroMask[{ {6,9}, 1}]:fill(1)
    expectedGradZeroMask[{ {7,9}, 6}]:fill(1)
    tester:assertTensorEq(gradZeroMask:float(), expectedGradZeroMask:float(), 1e-8)

    local gradInput = lm:backward({seq, seq_chars, char_to_ix}, gradOutput)
    tester:assertTensorSizeEq(gradInput[1], {opt.batch_size, opt.input_encoding_size})
    tester:asserteq(gradInput[2]:nElement(), 0, 'grad on seq should be empty tensor')

  end
  return f
end

-- test just the language model alone (without the criterion)
local function gradCheckLM()

  local dtype = 'torch.DoubleTensor'
  opt = {}
  opt.vocab_size = 10

  opt.char_vocab_size = 31
  opt.max_word_l = 10
  opt.feature_maps = '{50,100,150,200,200,200,200}'
  opt.kernels =  '{1,2,3,4,5,6,7}'
  opt.char_vec_size = 15
  opt.use_words = 0
  opt.use_chars = 1
  opt.highway_layers = 2
  opt.hsm = 0
  opt.batch_norm = 0
  opt.gpuid = -1
  loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
  loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes



  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModelCNN_Only(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)

  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 6 }] = 0
  local seq_chars = torch.LongTensor(opt.seq_length, opt.batch_size, opt.max_word_l):random(opt.char_vocab_size-2)

  -- evaluate the analytic gradient
  --debugger.enter()
  local output = lm:forward{seq, seq_chars, char_to_ix}
  local w = torch.randn(output:size(1), output:size(2), output:size(3))
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output, w))
  local gradOutput = w
  local gradInput, dummy = unpack(lm:backward({seq, seq_chars, char_to_ix}, gradOutput))

  -- create a loss function wrapper
  local function f()
    local output = lm:forward{seq, seq_chars, char_to_ix}
    local loss = torch.sum(torch.cmul(output, w))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, 1, 1e-6)

  print(gradInput)
  print(gradInput_num)
  debugger.enter()
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end

local function gradCheck()
  local dtype = 'torch.DoubleTensor'
  opt = {}
  opt.vocab_size = 10
  opt.char_vocab_size = 31
  opt.max_word_l = 10
  opt.feature_maps = '{50,100,150,200,200,200,200}'
  opt.kernels =  '{1,2,3,4,5,6,7}'
  opt.char_vec_size = 15
  opt.use_words = 0
  opt.use_chars = 1
  opt.highway_layers = 2
  opt.hsm = 0
  opt.batch_norm = 0
    opt.gpuid = -1
  loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
  loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes



  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModelCNN_Only(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  local seq_chars = torch.LongTensor(opt.seq_length, opt.batch_size, opt.max_word_l):random(opt.char_vocab_size-2)

  -- evaluate the analytic gradient
  local output = lm:forward{seq, seq_chars, char_to_ix}
  local loss = crit:forward(output, seq)
  local gradOutput = crit:backward(output, seq)
  local gradInput, dummy = unpack(lm:backward({seq, seq_chars, char_to_ix}, gradOutput))

  -- create a loss function wrapper
  local function f()
    local output = lm:forward{seq,seq_chars, char_to_ix}
    local loss = crit:forward(output, seq)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)
end

local function overfit()
  local dtype = 'torch.DoubleTensor'
  opt = {}
  opt.vocab_size = 10

  opt.char_vocab_size = 31
  opt.max_word_l = 10
  opt.feature_maps = '{50,100,150,200,200,200,200}'
  opt.kernels =  '{1,2,3,4,5,6,7}'
  opt.char_vec_size = 15
  opt.use_words = 0
  opt.use_chars = 1
  opt.highway_layers = 2
  opt.hsm = 0
  opt.batch_norm = 0
    opt.gpuid = -1
  loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
  loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes


  opt.input_encoding_size = 7
  opt.rnn_size = 24
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModelCNN_Only(opt)
  local crit = nn.LanguageModelCriterion()
  lm:type(dtype)
  crit:type(dtype)

  local seq = torch.LongTensor(opt.seq_length, opt.batch_size):random(opt.vocab_size)
  local seq_chars = torch.LongTensor(opt.seq_length, opt.batch_size, opt.max_word_l):fill(char_to_ix[' '])
  for i = 1,opt.seq_length do
      for j = 1, opt.batch_size do
          w = ix_to_word[tostring(seq[i][j])]
          seq_chars[i][j][1] = char_to_ix['{']
          local kk = 1
          for cc in w:gmatch(".") do
              seq_chars[i][j][kk +1] = char_to_ix[cc]
              kk = kk + 1
          end
          local last_idx = math.min(opt.max_word_l, kk + 2)
          seq_chars[i][j][last_idx] = char_to_ix['}']
      end
  end
  local params, grad_params = lm:getParameters()
  print('number of parameters:', params:nElement(), grad_params:nElement())
  local lstm_params = 4*(opt.input_encoding_size + opt.rnn_size)*opt.rnn_size + opt.rnn_size*4*2
  local output_params = opt.rnn_size * (opt.vocab_size + 1) + opt.vocab_size+1
  local table_params = (opt.vocab_size + 1) * opt.input_encoding_size
  local expected_params = lstm_params + output_params + table_params
  print('expected:', expected_params)

  local function lossFun()
    grad_params:zero()
    local output = lm:forward{seq, seq_chars, char_to_ix}
    local loss = crit:forward(output, seq)
    local gradOutput = crit:backward(output, seq)
    lm:backward({seq, seq_chars, char_to_ix}, gradOutput)
    return loss
  end

  local loss
  local grad_cache = grad_params:clone():fill(1e-8)
  print('trying to overfit the language model on toy data:')
  for t=1,100 do
    loss = lossFun()
    -- test that initial loss makes sense
    if t == 1 then tester:assertlt(math.abs(math.log(opt.vocab_size+1) - loss), 0.1) end
    grad_cache:addcmul(1, grad_params, grad_params)
    params:addcdiv(-1e-1, grad_params, torch.sqrt(grad_cache)) -- adagrad update
    print(string.format('iteration %d/30: loss %f', t, loss))
  end
  -- holy crap adagrad destroys the loss function!

  tester:assertlt(loss, 0.2)
end

-- check that we can call :sample() and that correct-looking things happen
local function sample()
  local dtype = 'torch.DoubleTensor'
  opt = {}
  opt.vocab_size = 10

  opt.char_vocab_size = 31
  opt.max_word_l = 10
  opt.feature_maps = '{50,100,150,200,200,200,200}'
  opt.kernels =  '{1,2,3,4,5,6,7}'
  opt.char_vec_size = 15
  opt.use_words = 0
  opt.use_chars = 1
  opt.highway_layers = 2
  opt.hsm = 0
  opt.batch_norm = 0
    opt.gpuid = -1
  loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
  loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes


  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 2
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModelCNN_Only(opt)

  local seq = lm:sample(char_to_ix, ix_to_word)

  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 1)
  tester:assertle(torch.max(seq), opt.vocab_size+1)
  print('\nsampled sequence:')
  print(seq)
end


-- check that we can call :sample_beam() and that correct-looking things happen
-- these are not very exhaustive tests and basic sanity checks
local function sample_beam()
  local dtype = 'torch.DoubleTensor'
  torch.manualSeed(1)

  opt = {}

  opt.char_vocab_size = 31
  opt.max_word_l = 10
  opt.feature_maps = '{50,100,150,200,200,200,200}'
  opt.kernels =  '{1,2,3,4,5,6,7}'
  opt.char_vec_size = 15
  opt.use_words = 0
  opt.use_chars = 1
  opt.highway_layers = 2
  opt.hsm = 0
  opt.batch_norm = 0
    opt.gpuid = -1
  loadstring('opt.kernels = ' .. opt.kernels)() -- get kernel sizes
  loadstring('opt.feature_maps = ' .. opt.feature_maps)() -- get feature map sizes


  opt.vocab_size = 10
  opt.input_encoding_size = 4
  opt.rnn_size = 8
  opt.num_layers = 1
  opt.dropout = 0
  opt.seq_length = 7
  opt.batch_size = 6
  local lm = nn.LanguageModelCNN_Only(opt)


  local seq_vanilla, logprobs_vanilla = lm:sample( char_to_ix, ix_to_word)
  local seq, logprobs = lm:sample(char_to_ix, ix_to_word, {beam_size = 1})

  -- check some basic I/O, types, etc.
  tester:assertTensorSizeEq(seq, {opt.seq_length, opt.batch_size})
  tester:asserteq(seq:type(), 'torch.LongTensor')
  tester:assertge(torch.min(seq), 0)
  tester:assertle(torch.max(seq), opt.vocab_size+1)

  -- doing beam search with beam size 1 should return exactly what we had before
  print('')
  print('vanilla sampling:')
  print(seq_vanilla)
  print('beam search sampling with beam size 1:')
  print(seq)
  tester:assertTensorEq(seq_vanilla, seq, 0) -- these are LongTensors, expect exact match
  tester:assertTensorEq(logprobs_vanilla, logprobs, 1e-6) -- logprobs too

  -- doing beam search with higher beam size should yield higher likelihood sequences
  local seq2, logprobs2 = lm:sample(char_to_ix, ix_to_word, {beam_size = 3})
  print(seq2)
  local logsum = torch.sum(logprobs, 1)
  local logsum2 = torch.sum(logprobs2, 1)
  print('')
  print('beam search sampling with beam size 1:')
  print(seq)
  print('beam search sampling with beam size 8:')
  print(seq2)
  print('logprobs:')
  print(logsum)
  print(logsum2)

  -- the logprobs should always be >=, since beam_search is better argmax inference
  tester:assert(torch.all(torch.gt(logsum2, logsum)))
end

--tests.doubleApiForwardTest = forwardApiTestFactory('torch.DoubleTensor')
--tests.floatApiForwardTest = forwardApiTestFactory('torch.FloatTensor')
--tests.cudaApiForwardTest = forwardApiTestFactory('torch.CudaTensor')
tests.gradCheck = gradCheck
--tests.gradCheckLM = gradCheckLM
--tests.overfit = overfit
--tests.sample = sample
--tests.sample_beam = sample_beam
--
tester:add(tests)
tester:run()

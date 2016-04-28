local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'
debugger = require('fb.debugger')

local LSTMTD = require 'model.LSTMTD'
local utils = require 'misc.utils'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------


local layer, parent = torch.class('nn.LanguageModelCNN_Only', 'nn.Module')

function layer:__init(opt)
    parent.__init(self)
    self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
    self.char_size = utils.getopt(opt, 'char_vocab_size')
    self.char_vec_size = utils.getopt(opt, 'char_vec_size')
    self.feature_maps = utils.getopt(opt, 'feature_maps')
    self.kernels = utils.getopt(opt,'kernels')
    self.max_word_l = utils.getopt(opt, 'max_word_l')
    self.use_words = utils.getopt(opt, 'use_words')
    self.use_chars = utils.getopt(opt, 'use_chars')
    self.batch_norm = utils.getopt(opt, 'batch_norm')
    self.highway_layers = utils.getopt(opt, 'highway_layers')
    self.hsm = utils.getopt(opt,'hsm')

    self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
    self.rnn_size = utils.getopt(opt, 'rnn_size')
    self.num_layers = utils.getopt(opt, 'num_layers', 1)
    local dropout = utils.getopt(opt, 'dropout', 0)
    -- options for Language Model
    self.seq_length = utils.getopt(opt, 'seq_length')
    -- create the core lstm network. note +1 for both the START and END tokens
    self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
    self.char_cnn = LSTMTD.lstmtdnn(self.input_encoding_size, self.vocab_size, self.rnn_size, self.char_size, self.char_vec_size, self.feature_maps, self.kernels, self.max_word_l, self.use_words, self.use_chars, self.batch_norm, self.highway_layers)
    self:_createInitState(1) -- will be lazily resized later during forward passes
end
function layer:getModulesList()
  return {self.core, self.char_cnn}
end


function layer:_createInitState(batch_size)
    assert(batch_size ~= nil, 'batch size must be provided')
    if not self.init_state then self.init_state = {} end -- lazy init
    for L=1, self.num_layers*2 do
        if self.init_state[L] then
            if self.init_state[L]:size(1) ~= batch_size then
                self.init_state[L]:resize(batch_size, self.rnn_size):zero() -- expand the memory
            end
        else
            self.init_state[L] = torch.zeros(batch_size, self.rnn_size)
        end
    end
    self.num_state = #self.init_state
end

function layer:get_input(x, x_char, t )
    local u = {}
    if opt.use_chars == 1 then table.insert(u, x_char[{{},t}]) end
    if opt.use_words == 1 then table.insert(u, x[{{},t}]) end
    return u
end


-- We assume, each sentence is started with a special start token.
function layer:forward(input)
    local seq = input[1] -- D * N (V_C = opt.char_vocab_size)
    local char_seq = input[2] -- D * N * V_C (V_C = opt.char_vocab_size)
    local char_to_ix = input[3]

    if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
    assert(seq:size(1) == self.seq_length)
    local batch_size = seq:size(2)
    -- Here we use self.seq_length + 1 instead of self.seq_length + 2. 
    -- This is because we do not add #start# token at the start of each sentence.
    --self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)
    self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1)
    self:_createInitState(batch_size)
    -- originally, return a (D + 2 ) * N * (M + 1) M = opt.vocab_size
    -- First, we need to forward the chars.
    ------------------ get minibatch -------------------
    --local x, y, x_char = loader:next_batch(1) --from train
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        seq = seq:float():cuda()
        char_seq = char_seq:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = self.init_state}
    local predictions = {}           -- softmax outputs

    self.lstm_inputs = {}
    local start_tok = "trats"
    -- encode the start token.
    local batch_start = torch.LongTensor(char_seq:size(2), char_seq:size(3)):fill(char_to_ix[' '])
    batch_start[{{}, 1}]:fill(char_to_ix['{'])
    local kk = 1
    for cc in start_tok:gmatch(".") do
        batch_start[{{}, kk+1}]:fill(char_to_ix[cc])
        kk = kk + 1
    end
    batch_start[{{}, kk+1}]:fill(char_to_ix['}'])

    self.clones.char_cnn[1]:training()
    local lst = self.clones['char_cnn'][1]:forward(batch_start)
    table.insert(self.lstm_inputs, lst)

    for t=1,self.seq_length do
        self.clones.char_cnn[t+1]:training() -- make sure we are in correct mode (this is cheap, sets flag)        
        local lst = self.clones['char_cnn'][t+1]:forward(char_seq[{t,{}}])
        table.insert(self.lstm_inputs, lst)
    end

    -- Now, it's time for lstm forward.
    -- different from the languagemodel.lua. We do not need to use LookUpTable to encode the input char sequence.
    if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
    self.state = {[0] = self.init_state}
    self.inputs = {}
    self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
    for t=1, self.seq_length + 1 do
        local can_skip = false
        local xt
        if t == 1 then
            xt = self.lstm_inputs[1]
        else
            -- feed in the rest of the sequence...
            if t >= 2 then
                local it = seq[t-1]:clone()
                if torch.sum(it) == 0 then
                    can_skip = true 
                end
            end

            if not can_skip then
                xt = self.lstm_inputs[t] -- just get the output from the CNN language model.
            end
        end

        if not can_skip then
            self.inputs[t] = {xt,unpack(self.state[t-1])}
            local out = self.clones['core'][t]:forward(self.inputs[t])
            self.output[t] = out[self.num_state+1] -- last element is the output vector
            self.state[t] = {} -- the rest is state
            for i=1,self.num_state do table.insert(self.state[t], out[i]) end
            self.tmax = t
        end
    end

    return self.output
end

function layer:backward(input, gradOutput)
    -- Firstly, we backward at the LSTM node.
    -- go backwards and lets compute gradients
    local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
    local seq = input[1] -- D * N * V_C (V_C = opt.char_vocab_size)
    local char_seq = input[2]
    local char_to_ix = input[3]
    
    for t = self.tmax, 1, -1 do
        -- concat state gradients and output vector gradients at time step t
        local dout = {}
        for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
        table.insert(dout, gradOutput[t])
        local dinputs = self.clones['core'][t]:backward(self.inputs[t], dout)
        -- split the gradient to xt and to state
        local dxt = dinputs[1] -- first element is the input vector
        dstate[t-1] = {} -- copy over rest to state grad
        for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
        -- continue backprop of xt
        if t == 1 then
            local start_tok = "trats"
            -- encode the start token.
            local batch_start = torch.LongTensor(char_seq:size(2), char_seq:size(3)):fill(char_to_ix[' '])
            batch_start[{{}, 1}]:fill(char_to_ix['{'])
            local kk = 1
            for cc in start_tok:gmatch(".") do
                batch_start[{{}, kk+1}]:fill(char_to_ix[cc])
                kk = kk + 1
            end
            batch_start[{{}, kk+1}]:fill(char_to_ix['}'])
            self.clones['char_cnn'][1]:backward(batch_start, dxt)
        else
            local it = char_seq[{t-1,{}}]
            --self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
            self.clones['char_cnn'][t]:backward(it, dxt)
        end
    end

    -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
    self.gradInput = {dimgs, torch.Tensor()}
    return self.gradInput
end


function layer:createClones()
    -- construct the net clones
    print('constructing clones inside the LanguageModel')
    self.clones = {}
    self.clones['char_cnn'] = {self.char_cnn}
    self.clones['core'] ={self.core}

    for t = 2, self.seq_length+2 do
        self.clones['core'][t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
        self.clones['char_cnn'][t] = self.char_cnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end


function layer:parameters()
    local p_core, g_core = self.core:parameters()
    local p_cnn, g_cnn = self.char_cnn:parameters()

    local params = {}
    for k,v in pairs(p_core) do table.insert(params, v) end
    for k,v in pairs(p_cnn) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g_core) do table.insert(grad_params, v) end
    for k,v in pairs(g_cnn) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:training()
    if self.clones == nil then self:createClones() end -- create these lazily if needed
    for i,n in pairs(self.clones) 
    do
        for k,v in pairs(n) do v:training() end
    end

end

function layer:evaluate()
    if self.clones == nil then self:createClones() end -- create these lazily if needed
    for i,n in pairs(self.clones) 
    do
        for k,v in pairs(n) do v:training() end
    end
end
--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(char_to_ix, ix_to_word, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  local start = utils.getopt(opt, 'start_char', '{')
  local end_ = utils.getopt(opt, 'end_char', '}')
  local zeropad = utils.getopt(opt, 'zeropad', ' ')

  if sample_max == 1 and beam_size > 1 then return self:sample_beam(char_to_ix, ix_to_word, opt) end -- indirection for beam search

  local batch_size = 2
  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step

  for t=1,self.seq_length+2 do

    local xt, it, sampleLogprobs
    if t == 1 then
      local start_tok = "trats"
      -- encode the start token.
      local batch_start = torch.LongTensor(batch_size, self.max_word_l):fill(char_to_ix[' '])
      batch_start[{{}, 1}]:fill(char_to_ix['{'])
      local kk = 1
      for cc in start_tok:gmatch(".") do
          batch_start[{{}, kk+1}]:fill(char_to_ix[cc])
          kk = kk + 1
      end
      batch_start[{{}, kk+1}]:fill(char_to_ix['}'])
      xt = self.char_cnn:forward(batch_start)
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      --xt = self.lookup_table:forward(it)
      -- it's kind of complicated.
      -- We need to encode this word. We also need to encode using cnn.

      local words = {}

      if #it:size() == 1 then
          for i_w = 1, it:size(1) do
              words[i_w] = ix_to_word[tostring(it[(i_w)])]
          end
      else
          for i_w = 1, it:size()[1] do
              words[i_w] = {}
              for j_w = 1, it:size()[2] do
                words[i_w][j_w] = ix_to_word[tostring(it[{i_w, j_w}])]
              end
          end
      end

      local chars_i
      if #it:size() == 1 then
          chars_i = torch.LongTensor( batch_size, self.max_word_l):fill(char_to_ix[' '])
      else
          chars_i = torch.LongTensor(it:size(2), batch_size, self.max_word_l):fill(char_to_ix[' '])
      end
      if #it:size() == 1 then
          for i_w, w_w in pairs(words) do
            local kk = 1
            chars_i[{i_w,1}] = char_to_ix[start]
            for cc in w_w:gmatch('.') do
                if kk+1 <= self.max_word_l then
                    chars_i[{i_w,kk+1}] = char_to_ix[cc]
                end
                kk = kk + 1
            end
            local last_idx = math.min(self.max_word_l, kk + 2)
            chars_i[{i_w, last_idx}] = char_to_ix[end_]
          end
      end    
      xt = self.char_cnn:forward(chars_i)
    end

    if t >= 2 then 
      seq[t-1] = it -- record the samples
      seqLogprobs[t-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end
    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
  end
  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
Implements beam search. Really tricky indexing stuff going on inside. 
Not 100% sure it's correct, and hard to fully unit test to satisfaction, but
it seems to work, doesn't crash, gives expected looking outputs, and seems to 
improve performance, so I am declaring this correct.
]]--
function layer:sample_beam(char_to_ix, ix_to_word, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size = 2
  local function compare(a,b) return a.p > b.p end -- used downstream
  local start = utils.getopt(opt, 'start_char', '{')
  local end_ = utils.getopt(opt, 'end_char', '}')
  local zeropad = utils.getopt(opt, 'zeropad', ' ')
  local word_to_ix = {}

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do

    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    for t=1,self.seq_length+1 do

      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
        -- Here, we should provide the #start token.
        -- feed in the start tokens
        local start_tok = "trats"
        -- encode the start token.
        --
        local batch_start = torch.LongTensor(beam_size, self.max_word_l):fill(char_to_ix[' '])
        batch_start[{{}, 1}]:fill(char_to_ix['{'])
        local kk = 1
        for cc in start_tok:gmatch(".") do
            batch_start[{{}, kk+1}]:fill(char_to_ix[cc])
            kk = kk + 1
        end
        batch_start[{{}, kk+1}]:fill(char_to_ix['}'])
        xt = self.char_cnn:forward(batch_start)

      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--

        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.max(beam_size,ys:size(2))
        local rows = beam_size
        if t == 3 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end

        local vix = 1
        local vix_cnt = 1
        while true do

          if vix_cnt > beam_size then
              break
          end

          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 2 then
            beam_seq[{ {1,t-2}, vix_cnt }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix_cnt }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix_cnt] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-1, vix_cnt }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix_cnt }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix_cnt] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+1 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            -- NOw, we need to back one.
            table.insert(done_beams, {seq = beam_seq[{ {}, vix_cnt }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix_cnt }]:clone(),
                                      p = beam_logprobs_sum[vix_cnt]
                                     })
            if v.c == self.vocab_size + 1 then
                vix_cnt = vix_cnt - 1
            end
          end
          vix = vix + 1
          vix_cnt = vix_cnt + 1
        end
        
        -- encode as vectors
        it = beam_seq[t-1]
        -- Now, we need to encode from word to char_cnn.
        local char_it = torch.LongTensor(it:size(1), self.max_word_l):fill(char_to_ix[' '])
        char_it[{{}, 1}]:fill(char_to_ix['{'])
        for i_it = 1, it:size(1) do
          w_i = ix_to_word[tostring(it[i_it])]
          local kk = 1
          for cc in w_i:gmatch(".") do
              char_it[{i_it, kk+1}] = char_to_ix[cc]
              kk = kk + 1
          end
          kk = math.min(self.max_word_l -1, kk)
          char_it[{i_it, kk+1}] = char_to_ix['}']
        end
        xt = self.char_cnn:forward(char_it)
      end
      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {xt,unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end
    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

require 'hdf5'
local utils = require 'misc.utils'

local DataLoaderCNN_LM = torch.class('DataLoaderCNN_LM')

function DataLoaderCNN_LM:__init(opt)
  
  -- load the json file which contains additional information about the dataset
  print('DataLoaderCNN_LM loading json file: ', opt.json_file)
  self.info = utils.read_json(opt.json_file)
  self.ix_to_word = self.info.ix_to_word
  self.ix_to_char = self.info.ix_to_char
  self.char_to_ix = self.info.char_to_ix
  self.splits = self.info.splits

  self.vocab_size = utils.count_keys(self.ix_to_word)
  self.char_vocab_size = utils.count_keys(self.ix_to_char)
  print('vocab size is ' .. self.vocab_size)
  print('char vocab size is ' .. self.char_vocab_size)
  
  -- open the hdf5 file
  print('DataLoaderCNN_LM loading h5 file: ', opt.h5_file)
  self.h5_file = hdf5.open(opt.h5_file, 'r')
  
  -- load in the sequence data
  local seq_size = self.h5_file:read('/labels'):dataspaceSize()
  local char_seq_size = self.h5_file:read('/chars'):dataspaceSize()
  self.seq_length = seq_size[2]
  self.max_word_l = char_seq_size[3]
  print('max sequence length in data is ' .. self.seq_length)
  print('max word length in data is ' .. self.max_word_l)
  -- load the pointers in full to RAM (should be small enough)
  
  -- separate out indexes for each of the provided splits
  self.split_ix = {}
  self.iterators = {}
  for i,split in pairs(self.splits) do
    if not self.split_ix[split] then
      -- initialize new split
      self.split_ix[split] = {}
      self.iterators[split] = 1
    end
    table.insert(self.split_ix[split], i)
  end
  for k,v in pairs(self.split_ix) do
    print(string.format('assigned %d to split %s', #v, k))
  end
end

function DataLoaderCNN_LM:resetIterator(split)
  self.iterators[split] = 1
end

function DataLoaderCNN_LM:getVocabSize()
  return self.vocab_size
end

function DataLoaderCNN_LM:getCharVocabSize()
  return self.char_vocab_size
end

function DataLoaderCNN_LM:getMaxWordL()
  return self.max_word_l
end

function DataLoaderCNN_LM:getCharVocab()
  return self.ix_to_char
end


function DataLoaderCNN_LM:getVocab()
  return self.ix_to_word
end

function DataLoaderCNN_LM:getSeqLength()
  return self.seq_length
end

--[[
  Split is a string identifier (e.g. train|val|test)
  Returns a batch of data:
  - y (L,M) containing the captions as columns (which is better for contiguous memory during training)
  - info table of length N, containing additional information
  The data is iterated linearly in order. Iterators for any split can be reset manually with resetIterator()
--]]
function DataLoaderCNN_LM:getBatch(opt)
  local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
  local batch_size = utils.getopt(opt, 'batch_size', 5)
  local split_ix = self.split_ix[split]
  assert(split_ix, 'split ' .. split .. ' not found.')

  local label_batch = torch.LongTensor(batch_size, self.seq_length)
  local char_batch = torch.LongTensor(batch_size, self.seq_length, self.max_word_l)
  local max_index = #split_ix
  local wrapped = false
  for i=1,batch_size do

    local ri = self.iterators[split] -- get next index from iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterators[split] = ri_next
    ix = split_ix[ri]
    assert(ix ~= nil, 'bug: split ' .. split .. ' was accessed out of bounds with ' .. ri)


    local seq = self.h5_file:read('/labels'):partial({ix, ix}, {1,self.seq_length})
    local char_seq = self.h5_file:read('/chars'):partial({ix, ix}, {1,self.seq_length}, {1,self.max_word_l})
    label_batch[{ {i,i} }] = seq
    char_batch[{ {i,i},{} }] = char_seq

  end

  local data = {}
  data.labels = label_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.chars = char_batch:transpose(1,2):contiguous() -- note: make label sequences go down as columns
  data.bounds = {it_pos_now = self.iterators[split], it_max = #split_ix, wrapped = wrapped}
  return data
end


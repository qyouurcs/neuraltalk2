import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
from scipy.misc import imread, imresize
import pdb

def prepro_sentences(fn):
  sents = []
  with open(fn, 'r') as fid:
    for i,s in enumerate(fid):
      #txt = str(s).lower().translate(None, string.punctuation).strip().split()
      txt = str(s).lower().strip().split()
      sents.append(txt)
      if i < 10:
        print txt
  return sents

def build_vocab(sents, params):
  count_thr = params['word_count_threshold']
  # count up the number of words
  counts = {}
  for txt in sents:
    for w in txt:
      counts[w] = counts.get(w, 0) + 1
  cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
  print 'top words and their counts:'
  print '\n'.join(map(str,cw[:20]))

  # print some stats
  total_words = sum(counts.itervalues())
  print 'total words:', total_words
  bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
  vocab = [w for w,n in counts.iteritems() if n > count_thr]
  bad_count = sum(counts[w] for w in bad_words)
  print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
  print 'number of words in vocab would be %d' % (len(vocab), )
  print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

  # lets look at the distribution of lengths as well
  sent_lengths = {}
  for txt in sents:
    nw = len(txt)
    sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
  max_len = max(sent_lengths.keys())
  print 'max length sentence in raw data: ', max_len
  print 'sentence length distribution (count, number of words):'
  sum_len = sum(sent_lengths.values())
  for i in xrange(max_len+1):
    print '%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len)

  # lets now produce the final annotations
  if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
  final_sents = []
  for txt in sents:
    sent = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
    final_sents.append(sent)

  return vocab, final_sents

def assign_splits(final_sents, params):
  num_val = params['num_val']
  num_test = params['num_test']
  splits = []
  for i,sent in enumerate(final_sents):
    if i < num_val:
      splits.append('val')
    elif i < num_val + num_test: 
      splits.append('test')
    else: 
      splits.append('train')
  print 'assigned %d to val, %d to test.' % (num_val, num_test)
  return splits

def encode_captions(final_sents, params, wtoi):
  """ 
  encode all captions into one large array, which will be 1-indexed.
  also produces label_start_ix and label_end_ix which store 1-indexed 
  and inclusive (Lua-style) pointers to the first and last caption for
  each image in the dataset.
  """

  max_length = params['max_length']
  start = params['start_char']
  end = params['end_char']
  zeropad = params['zeropad']
  max_word_l = params['max_word_l']

  idx2char = {1:zeropad, 2:start, 3:end}
  char2idx = {zeropad:1, start:2, end:3}

  N = len(final_sents)

  label_arrays = np.zeros((N, max_length + 1), dtype = 'uint32')
  label_chars_array = np.zeros((N, max_length + 1, max_word_l), dtype = 'uint32')
  for i,sent in enumerate(final_sents):
    Li = np.zeros((1,max_length + 1), dtype='uint32')
    Li[0,0] = wtoi['trats']
    chars_i = np.zeros((1,max_length + 1, max_word_l), dtype='uint32')
    chars_i[:] = char2idx[zeropad]
    
    chars_i[0,0,0] = char2idx[start]
    kk = 0
    for kk, cc in enumerate('trats'):
      if cc not in char2idx:
        char2idx[cc] = len(char2idx) + 1
        idx2char[len(idx2char)+1] = cc
      if kk + 1 < max_word_l:
        chars_i[0,0,kk + 1] = char2idx[cc]

    last_idx = min(max_word_l-1, kk + 2)
    chars_i[0,0,last_idx] = char2idx[end]
    for k,w in enumerate(sent):
      if k < max_length:
        Li[0,k + 1] = wtoi[w]
        chars_i[0,k+1,0] = char2idx[start]
        kk = 0
        for kk, cc in enumerate(w): 
          if cc not in char2idx:
            char2idx[cc] = len(char2idx)+1
            idx2char[len(idx2char)+1] = cc
          if kk + 1 < max_word_l:
            chars_i[0,k + 1,kk + 1] = char2idx[cc]
        last_idx = min(max_word_l-1, kk + 2)
        chars_i[0,k + 1,last_idx] = char2idx[end]

    # note: word indices are 1-indexed, and captions are padded with zeros
    label_arrays[i,:] = Li[0,:]
    label_chars_array[i,:] = chars_i[0,:]
  
  print "L", label_arrays.shape
  print "C", label_chars_array.shape

  print 'encoded captions to array of size ', `label_arrays.shape`
  return label_arrays, label_chars_array, idx2char, char2idx

def main(params):
  
  sent_fn = params['txt_fn']
  sents = prepro_sentences(sent_fn)

  # create the vocab
  vocab, final_sents = build_vocab(sents,params)
  # Now, we add the start to the vocab.
  itow = {i+2:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
  wtoi = {w:i+2 for i,w in enumerate(vocab)} # inverse table

  itow[1] = "trats"
  wtoi["trats"] = 1

  # assign the splits
  splits = assign_splits(final_sents, params)
  
  # encode captions in large arrays, ready to ship to hdf5 file
  L, C, idx2char, char2idx = encode_captions(final_sents, params, wtoi)

  # create output h5 file
  N = len(final_sents)
  f = h5py.File(params['output_h5'], "w")
  f.create_dataset("labels", dtype='uint32', data=L)
  f.create_dataset("chars", dtype='uint32', data=C)
  f.close()

  # create output json file
  out = {}
  out['ix_to_word'] = itow # encode the (1-indexed) vocab
  out['ix_to_char'] = idx2char# encode the (1-indexed) vocab
  out['char_to_ix'] = char2idx# encode the (1-indexed) vocab
  out['splits'] = splits
  json.dump(out, open(params['output_json'], 'w'))
  print 'wrote ', params['output_json']

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--txt_fn', required=True, help='input sentence file to process')
  parser.add_argument('--num_val', required=True, type=int, help='number of sents to assign to validation data (for CV etc)')
  parser.add_argument('--output_json', default='data.json', help='output json file')
  parser.add_argument('--output_h5', default='data.h5', help='output h5 file')
  
  # options
  parser.add_argument('--max_length', default=50, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
  parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
  parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
  parser.add_argument('--start_char', default='{', help='start-of-word token')
  parser.add_argument('--end_char', default='}', help='start-of-word token')
  parser.add_argument('--zeropad', default=' ', help='zero-pad token')
  parser.add_argument('--max_word_l', default=65, type=int, help='zero-pad token')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed input parameters:'
  print json.dumps(params, indent = 2)
  main(params)

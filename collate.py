import torch
from torch.nn.utils.rnn import pad_sequence

def speech_collate_fn(batch,pad_idx=0):
  waveforms,labels=zip(*batch)
  waveform_lengths=torch.tensor([w.shape[0] for w in waveforms],dtype=torch.long)
  label_lengths=torch.tensor([len(l) for l in labels],dtype=torch.long)

  padded_waveforms=pad_sequence(waveforms,batch_first=True)
  padded_labels=pad_sequence(labels,batch_first=True,padding_value=pad_idx)

  return padded_waveforms,waveform_lengths,padded_labels,label_lengths
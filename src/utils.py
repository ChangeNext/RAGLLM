import shutil
import torch

def save_checkpoint(state, is_best, filename='checkpoint'):
  torch.save(state, filename + '.pth.tar')
  if is_best:
    shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')
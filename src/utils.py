import shutil
import torch

def save_checkpoint(state, current, logger, filename='checkpoint'):
  torch.save(state, filename + '.pth.tar')
  if current == "best":
    logger.info("save best model")
    shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')
  elif current == "t2m":
    logger.info("save best t2m model")
    shutil.copyfile(filename + '.pth.tar', filename + '_best_t2m.pth.tar')
  elif current == "m2t":
    logger.info("save best m2t model")
    shutil.copyfile(filename + '.pth.tar', filename + '_best_m2t.pth.tar')
    
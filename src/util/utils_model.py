import numpy as np 
import torch
import torch.optim as optim
import logging
import os 
import sys 

def getCi(accLog):

    mean = np.mean(accLog)
    std = np.std(accLog)
    ci95 = 1.96*std/np.sqrt(len(accLog))

    return mean, ci95

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger
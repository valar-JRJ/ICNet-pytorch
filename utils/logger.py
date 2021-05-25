# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import datetime

__all__ = ['SetupLogger', 'get_logger']

# reference from: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
def SetupLogger(name, save_dir, distributed_rank, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

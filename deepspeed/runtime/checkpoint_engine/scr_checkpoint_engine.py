'''Copyright The Microsoft DeepSpeed Team'''

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine

import scr


class SCRCheckpointEngine(CheckpointEngine):
    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, tag):
        log_dist(f"[SCR] Checkpoint {tag} is starting to save!", ranks=[0])
        scr.start_output(tag, scr.FLAG_CHECKPOINT | scr.FLAG_OUTPUT)

    def makedirs(self, path, exist_ok=False):
        # SCR delays creating directories until it flushes the checkpoint.
        # Based on how the user has configured their run,
        # SCR may discard some checkpoints without ever flushing them.
        pass

    def save(self, state_dict, path: str):
        path = scr.route_file(path)
        torch.save(state_dict, path)

    def load(self, path: str, map_location=None):
        path = scr.route_file(path)
        partition = torch.load(path, map_location=map_location)
        return partition

    def commit(self, tag):
        scr.complete_output(True)
        logger.info(f"[SCR] Checkpoint {tag} is complete!", ranks=[0])
        return True

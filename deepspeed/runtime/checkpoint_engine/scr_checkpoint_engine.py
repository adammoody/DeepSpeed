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
        log_dist(f"[Torch] Checkpoint {tag} is begin to save!", ranks=[0])
        scr.start_output(tag, scr.FLAG_CHECKPOINT | scr.FLAG_OUTPUT)

    def save(self, state_dict, path: str):
        path = scr.route_file(path)
        torch.save(state_dict, path)
        return None

    def load(self, path: str, map_location=None):
        path = scr.route_file(path)
        partition = torch.load(path, map_location=map_location)
        return partition

    def commit(self, tag):
        scr.complete_output(True)
        logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True

"""
DECLOUD Creator Kit
===================

Create and manage federated learning rounds on DECLOUD.

Usage:
    from decloud_creator import Creator
    
    creator = Creator()
    creator.login("your_private_key")
    
    round_id = creator.create_round(
        model="model.pt",
        dataset="Cifar10", 
        reward=0.01
    )
    
    creator.watch(round_id)
"""

from .config import Config, DATASETS, POPULAR_DATASETS
from .creator import Creator, RoundInfo
from .storage import RoundStorage, StoredRound
from .ipfs import IPFSUploader

__version__ = "0.1.0"
__all__ = [
    "Creator",
    "Config", 
    "RoundInfo",
    "RoundStorage",
    "StoredRound",
    "IPFSUploader",
    "DATASETS",
    "POPULAR_DATASETS",
]

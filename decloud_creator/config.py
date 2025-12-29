"""
DECLOUD Creator Configuration
"""

import os
import json
from typing import Optional, List
from dataclasses import dataclass, asdict


NETWORKS = {
    "devnet": {
        "rpc_url": "https://api.devnet.solana.com",
        "ws_url": "wss://api.devnet.solana.com",
    },
    "mainnet-beta": {
        "rpc_url": "https://api.mainnet-beta.solana.com",
        "ws_url": "wss://api.mainnet-beta.solana.com",
    },
}

DEFAULT_PROGRAM_ID = "HvQ8c3BBCsibJpH74UDXbvqEidJymzFYyGjNjRn7MYwC"


@dataclass
class Config:
    """Creator configuration"""
    
    network: str = "mainnet-beta"
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    ws_url: str = "wss://api.mainnet-beta.solana.com"
    program_id: str = DEFAULT_PROGRAM_ID
    
    # Paths
    data_dir: str = "~/.decloud-creator"
    idl_path: str = ""
    
    # Auth
    private_key: Optional[str] = None
    
    # IPFS
    ipfs_gateway: str = "https://ipfs.io/ipfs/"
    pinata_api_key: Optional[str] = None
    pinata_secret_key: Optional[str] = None
    
    # Defaults
    default_reward: float = 0.01  # SOL
    
    def __post_init__(self):
        self.data_dir = os.path.expanduser(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        
        if not self.idl_path:
            self.idl_path = self._find_idl()
        self.idl_path = self.idl_path.replace("\\", "/") 
    
    def _find_idl(self) -> str:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        kit_dir = os.path.dirname(package_dir)
        
        search_paths = [
            os.path.join(kit_dir, "idl.json"),
            os.path.join(package_dir, "idl.json"),
            os.path.join(os.getcwd(), "idl.json"),
            os.path.join(os.getcwd(), "target", "idl", "federated_ai.json"),
            os.path.expanduser("~/.decloud/idl.json"),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return os.path.join(kit_dir, "idl.json")
    
    @classmethod
    def load(cls) -> "Config":
        config_path = os.path.expanduser("~/.decloud-creator/config.json")
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                data = json.load(f)
                return cls(**data)
        
        return cls()
    
    def save(self):
        config_path = os.path.join(self.data_dir, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        data = {k: v for k, v in asdict(self).items() if v is not None}
        
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)


# Supported datasets (must match lib.rs enum)
DATASETS = [
    # Images - Classification
    "Cifar10", "Cifar100", "Mnist", "FashionMnist", "Emnist", "Kmnist",
    "Food101", "Flowers102", "StanfordDogs", "StanfordCars", "OxfordPets",
    "CatsVsDogs", "Eurosat", "Svhn", "Caltech101", "Caltech256",
    
    # Text - Sentiment
    "Imdb", "Sst2", "Sst5", "YelpReviews", "AmazonPolarity",
    "RottenTomatoes", "FinancialSentiment", "TweetSentiment",
    
    # Text - Classification
    "AgNews", "Dbpedia", "YahooAnswers", "TwentyNewsgroups",
    
    # Text - Spam & Toxicity
    "SmsSpam", "HateSpeech", "CivilComments", "Toxicity",
    
    # And more...
    "Custom",
]

POPULAR_DATASETS = [
    "Cifar10", "Cifar100", "Mnist", "FashionMnist", 
    "Imdb", "Sst2", "AgNews",
]
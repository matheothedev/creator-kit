"""
Configuration for Decloud Creator Kit
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict

# Paths
HOME_DIR = Path.home()
CONFIG_DIR = HOME_DIR / ".decloud-creator"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHE_DIR = CONFIG_DIR / "cache"
PACKAGES_DIR = CONFIG_DIR / "packages"
MODELS_DIR = CONFIG_DIR / "models"
DATA_DIR = CONFIG_DIR / "data"

# Solana
PROGRAM_ID = "DCLDgP6xHuVmcKuGvAzKEkrbSYHApp9568JhVoXsF2Hh"
TREASURY = "FzuCxi65QyFXAGbHcXB28RXqyBZSZ5KXLQxeofx1P9K2"

# RPC Endpoints
RPC_ENDPOINTS = {
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
    "testnet": "https://api.testnet.solana.com",
}

# IPFS Gateways (Lighthouse primary)
IPFS_GATEWAYS = [
    "https://gateway.lighthouse.storage/ipfs/",
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
]

# Datasets enum (must match Solana program)
DATASETS = {
    "Cifar10": 0, "Cifar100": 1, "Mnist": 2, "FashionMnist": 3, "Emnist": 4,
    "Kmnist": 5, "Food101": 6, "Flowers102": 7, "StanfordDogs": 8, "StanfordCars": 9,
    "OxfordPets": 10, "CatsVsDogs": 11, "Eurosat": 12, "Svhn": 13, "Caltech101": 14,
    "Caltech256": 15, "Imdb": 16, "Sst2": 17, "Sst5": 18, "YelpReviews": 19,
    "AmazonPolarity": 20, "RottenTomatoes": 21, "FinancialSentiment": 22, "TweetSentiment": 23,
    "AgNews": 24, "Dbpedia": 25, "YahooAnswers": 26, "TwentyNewsgroups": 27,
    "SmsSpam": 28, "HateSpeech": 29, "CivilComments": 30, "Toxicity": 31,
    "ClincIntent": 32, "Banking77": 33, "SnipsIntent": 34, "Conll2003": 35,
    "Wnut17": 36, "Squad": 37, "SquadV2": 38, "TriviaQa": 39, "BoolQ": 40,
    "CommonsenseQa": 41, "Stsb": 42, "Mrpc": 43, "Qqp": 44, "Snli": 45,
    "Mnli": 46, "CnnDailymail": 47, "Xsum": 48, "Samsum": 49, "SpeechCommands": 50,
    "Librispeech": 51, "CommonVoice": 52, "Gtzan": 53, "Esc50": 54, "Urbansound8k": 55,
    "Nsynth": 56, "Ravdess": 57, "CremaD": 58, "Iemocap": 59, "Iris": 60,
    "Wine": 61, "Diabetes": 62, "BreastCancer": 63, "CaliforniaHousing": 64,
    "AdultIncome": 65, "BankMarketing": 66, "CreditDefault": 67, "Titanic": 68,
    "HeartDisease": 69, "ChestXray": 70, "SkinCancer": 71, "DiabeticRetinopathy": 72,
    "BrainTumor": 73, "Malaria": 74, "BloodCells": 75, "CovidXray": 76,
    "PubmedQa": 77, "MedQa": 78, "Electricity": 79, "Weather": 80, "StockPrices": 81,
    "EcgHeartbeat": 82, "CodeSearchNet": 83, "Humaneval": 84, "Mbpp": 85,
    "Spider": 86, "Cora": 87, "Citeseer": 88, "Qm9": 89, "NslKdd": 90,
    "CreditCardFraud": 91, "Phishing": 92, "Movielens1m": 93, "Movielens100k": 94,
    "Xnli": 95, "AmazonReviewsMulti": 96, "Sberquad": 97,
}

DATASET_ID_TO_NAME = {v: k for k, v in DATASETS.items()}

# Head params ratio
HEAD_PARAMS_RATIO = 0.15


class Config:
    """Creator configuration"""
    
    def __init__(self):
        # Wallet
        self.private_key: Optional[str] = None
        self.network: str = "devnet"
        self.custom_rpc: Optional[str] = None  # Custom RPC URL (overrides network default)
        
        # Lighthouse Storage (IPFS)
        self.lighthouse_api_key: Optional[str] = None
        
        # Model building
        self.head_ratio: float = HEAD_PARAMS_RATIO
        self.embedding_limit: int = 10000
        self.batch_size: int = 64
        
        # Created rounds tracking
        self.created_rounds: Dict[int, str] = {}  # round_id -> model_cid
        
        self._ensure_dirs()
        self._load()
    
    def _ensure_dirs(self):
        """Create directories"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load(self):
        """Load config from file"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.private_key = data.get("private_key")
                self.network = data.get("network", "devnet")
                self.custom_rpc = data.get("custom_rpc")
                self.lighthouse_api_key = data.get("lighthouse_api_key")
                self.head_ratio = data.get("head_ratio", HEAD_PARAMS_RATIO)
                self.embedding_limit = data.get("embedding_limit", 10000)
                self.batch_size = data.get("batch_size", 64)
                self.created_rounds = data.get("created_rounds", {})
    
    def save(self):
        """Save config"""
        data = {
            "private_key": self.private_key,
            "network": self.network,
            "custom_rpc": self.custom_rpc,
            "lighthouse_api_key": self.lighthouse_api_key,
            "head_ratio": self.head_ratio,
            "embedding_limit": self.embedding_limit,
            "batch_size": self.batch_size,
            "created_rounds": self.created_rounds,
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    @property
    def rpc_url(self) -> str:
        if self.custom_rpc:
            return self.custom_rpc
        return RPC_ENDPOINTS.get(self.network, RPC_ENDPOINTS["devnet"])
    
    def has_lighthouse(self) -> bool:
        """Check if Lighthouse API key is configured"""
        return bool(self.lighthouse_api_key)
    
    def track_round(self, round_id: int, model_cid: str):
        """Track created round"""
        self.created_rounds[str(round_id)] = model_cid
        self.save()


config = Config()

"""
Decloud Creator Kit
Creates model packages for training rounds

Automatically splits model: last 15% of parameters = head (trainable)

Usage:
    python creator_kit.py --model model.pt --dataset Cifar10 --output ./package
    python creator_kit.py  # interactive mode
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import save_file as save_safetensors
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.prompt import Prompt, Confirm

console = Console()

# Head = last X% of parameters
HEAD_PARAMS_RATIO = 0.15

# Supported layer types
LAYER_TYPE_MAP = {
    nn.Linear: "Linear",
    nn.Conv1d: "Conv1d",
    nn.Conv2d: "Conv2d",
    nn.BatchNorm1d: "BatchNorm1d",
    nn.BatchNorm2d: "BatchNorm2d",
    nn.LayerNorm: "LayerNorm",
    nn.ReLU: "ReLU",
    nn.GELU: "GELU",
    nn.SiLU: "SiLU",
    nn.Tanh: "Tanh",
    nn.Sigmoid: "Sigmoid",
    nn.Softmax: "Softmax",
    nn.Dropout: "Dropout",
    nn.Dropout2d: "Dropout2d",
    nn.MaxPool1d: "MaxPool1d",
    nn.MaxPool2d: "MaxPool2d",
    nn.AvgPool1d: "AvgPool1d",
    nn.AvgPool2d: "AvgPool2d",
    nn.AdaptiveAvgPool1d: "AdaptiveAvgPool1d",
    nn.AdaptiveAvgPool2d: "AdaptiveAvgPool2d",
    nn.Flatten: "Flatten",
}

DATASETS = [
    # Image Classification
    "Cifar10", "Cifar100", "Mnist", "FashionMnist", "Emnist", "Kmnist",
    "Food101", "Flowers102", "StanfordDogs", "StanfordCars",
    "OxfordPets", "CatsVsDogs", "Eurosat", "Svhn", "Caltech101", "Caltech256",
    # Text Classification
    "Imdb", "Sst2", "Sst5", "YelpReviews", "AmazonPolarity", "RottenTomatoes",
    "FinancialSentiment", "TweetSentiment", "AgNews", "Dbpedia", "YahooAnswers",
    "TwentyNewsgroups", "SmsSpam", "HateSpeech", "CivilComments", "Toxicity",
    # Intent/NER
    "ClincIntent", "Banking77", "SnipsIntent", "Conll2003", "Wnut17",
    # QA
    "Squad", "SquadV2", "TriviaQa", "BoolQ", "CommonsenseQa",
    # Similarity
    "Stsb", "Mrpc", "Qqp", "Snli", "Mnli",
    # Summarization
    "CnnDailymail", "Xsum", "Samsum",
    # Audio
    "SpeechCommands", "Librispeech", "CommonVoice", "Gtzan", "Esc50",
    "Urbansound8k", "Nsynth", "Ravdess", "CremaD", "Iemocap",
    # Tabular
    "Iris", "Wine", "Diabetes", "BreastCancer", "CaliforniaHousing",
    "AdultIncome", "BankMarketing", "CreditDefault", "Titanic", "HeartDisease",
    # Medical
    "ChestXray", "SkinCancer", "DiabeticRetinopathy", "BrainTumor", "Malaria",
    "BloodCells", "CovidXray", "PubmedQa", "MedQa",
    # Time Series
    "Electricity", "Weather", "StockPrices", "EcgHeartbeat",
    # Code
    "CodeSearchNet", "Humaneval", "Mbpp", "Spider",
    # Graph
    "Cora", "Citeseer", "Qm9",
    # Security
    "NslKdd", "CreditCardFraud", "Phishing",
    # Recommender
    "Movielens1m", "Movielens100k",
    # Multilingual
    "Xnli", "AmazonReviewsMulti", "Sberquad",
]


@dataclass
class LayerInfo:
    """Info about a layer"""
    name: str
    type_name: str
    module: nn.Module
    num_params: int
    params: Dict[str, Any]


def get_layer_params(module: nn.Module) -> Dict[str, Any]:
    """Extract layer config params"""
    params = {}
    
    if isinstance(module, nn.Linear):
        params = {"in_features": module.in_features, "out_features": module.out_features}
        if module.bias is None:
            params["bias"] = False
            
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        params = {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size[0] if len(set(module.kernel_size)) == 1 else list(module.kernel_size),
            "stride": module.stride[0] if len(set(module.stride)) == 1 else list(module.stride),
            "padding": module.padding[0] if len(set(module.padding)) == 1 else list(module.padding),
        }
        if module.bias is None:
            params["bias"] = False
            
    elif isinstance(module, nn.BatchNorm1d):
        params = {"num_features": module.num_features}
        
    elif isinstance(module, nn.BatchNorm2d):
        params = {"num_features": module.num_features}
        
    elif isinstance(module, nn.LayerNorm):
        shape = module.normalized_shape
        params = {"normalized_shape": shape[0] if len(shape) == 1 else list(shape)}
        
    elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
        params = {"p": module.p}
        
    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        params = {
            "kernel_size": module.kernel_size if isinstance(module.kernel_size, int) else list(module.kernel_size),
            "stride": module.stride if isinstance(module.stride, int) else list(module.stride),
        }
        
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        out = module.output_size
        params = {"output_size": out if isinstance(out, int) else list(out)}
        
    elif isinstance(module, nn.Flatten):
        params = {"start_dim": module.start_dim, "end_dim": module.end_dim}
        
    elif isinstance(module, nn.Softmax):
        params = {"dim": module.dim}
    
    return params


def extract_layers(model: nn.Module) -> List[LayerInfo]:
    """Extract all leaf layers from model"""
    layers = []
    
    for name, module in model.named_modules():
        # Skip containers
        if len(list(module.children())) > 0:
            continue
        
        # Skip root
        if name == "":
            continue
        
        module_type = type(module)
        num_params = sum(p.numel() for p in module.parameters())
        
        if module_type in LAYER_TYPE_MAP:
            type_name = LAYER_TYPE_MAP[module_type]
        else:
            type_name = f"Unknown:{module_type.__name__}"
        
        layers.append(LayerInfo(
            name=name,
            type_name=type_name,
            module=module,
            num_params=num_params,
            params=get_layer_params(module),
        ))
    
    return layers


def find_split_index(layers: List[LayerInfo], ratio: float = HEAD_PARAMS_RATIO) -> int:
    """
    Find split index where last `ratio` of parameters become head.
    Returns index of first head layer.
    """
    total_params = sum(l.num_params for l in layers)
    target_head_params = total_params * ratio
    
    accumulated = 0
    split_idx = len(layers)
    
    # Go backwards
    for i in range(len(layers) - 1, -1, -1):
        accumulated += layers[i].num_params
        split_idx = i
        if accumulated >= target_head_params:
            break
    
    return split_idx


def display_split(layers: List[LayerInfo], split_idx: int):
    """Display model split"""
    total_params = sum(l.num_params for l in layers)
    encoder_params = sum(l.num_params for l in layers[:split_idx])
    head_params = sum(l.num_params for l in layers[split_idx:])
    
    table = Table(title="Model Split")
    table.add_column("#", style="dim", width=4)
    table.add_column("Part", style="bold", width=8)
    table.add_column("Name", style="yellow")
    table.add_column("Type", style="cyan")
    table.add_column("Params", style="green", justify="right")
    
    for i, layer in enumerate(layers):
        part = "[blue]ENCODER[/blue]" if i < split_idx else "[red]HEAD[/red]"
        params_str = f"{layer.num_params:,}" if layer.num_params > 0 else "-"
        table.add_row(str(i), part, layer.name[:30], layer.type_name, params_str)
    
    console.print(table)
    console.print(f"\n[dim]Total: {total_params:,} params[/dim]")
    console.print(f"[blue]Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)[/blue]")
    console.print(f"[red]Head: {head_params:,} ({head_params/total_params*100:.1f}%)[/red]")


def build_encoder(model: nn.Module, layers: List[LayerInfo], split_idx: int) -> nn.Module:
    """Build encoder from layers"""
    encoder_layer_names = {l.name for l in layers[:split_idx]}
    
    # Try to find common patterns
    # Option 1: model has .features / .encoder / .backbone
    for attr in ["features", "encoder", "backbone", "base"]:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            # Check if all encoder layers are in this
            candidate_names = {n for n, _ in candidate.named_modules()}
            if encoder_layer_names.issubset(candidate_names) or len(candidate_names) > 0:
                return candidate
    
    # Option 2: Build sequential from modules
    modules = []
    for layer in layers[:split_idx]:
        modules.append(layer.module)
    
    return nn.Sequential(*modules)


def build_head_config(layers: List[LayerInfo]) -> Dict[str, Any]:
    """Build config.json for head"""
    config_layers = []
    
    for layer in layers:
        if layer.type_name.startswith("Unknown"):
            console.print(f"[yellow]⚠ Skipping unknown layer: {layer.name} ({layer.type_name})[/yellow]")
            continue
        
        config_layers.append({
            "type": layer.type_name,
            "params": layer.params,
        })
    
    return {"layers": config_layers}


def extract_head_weights(layers: List[LayerInfo]) -> Dict[str, torch.Tensor]:
    """Extract weights from head layers"""
    weights = {}
    layer_idx = 0
    
    for layer in layers:
        if layer.type_name.startswith("Unknown"):
            continue
        
        state = layer.module.state_dict()
        for key, tensor in state.items():
            weights[f"layers.{layer_idx}.{key}"] = tensor.clone()
        
        if state:
            layer_idx += 1
    
    return weights


def load_dataset(name: str, limit: int = 10000):
    """Load test dataset from torchvision or huggingface"""
    import torchvision
    import torchvision.transforms as T
    
    data_root = Path.home() / ".decloud-creator" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    t_gray = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    t_rgb = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    t_resize_224 = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    t_resize_32 = T.Compose([
        T.Resize((32, 32)), T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = None
    
    # ═══════════════════════════════════════════════════════════════
    # TORCHVISION DATASETS
    # ═══════════════════════════════════════════════════════════════
    
    if name == "Cifar10":
        dataset = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=t_rgb)
    elif name == "Cifar100":
        dataset = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=t_rgb)
    elif name == "Mnist":
        dataset = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=t_gray)
    elif name == "FashionMnist":
        dataset = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=t_gray)
    elif name == "Emnist":
        dataset = torchvision.datasets.EMNIST(data_root, split="balanced", train=False, download=True, transform=t_gray)
    elif name == "Kmnist":
        dataset = torchvision.datasets.KMNIST(data_root, train=False, download=True, transform=t_gray)
    elif name == "Svhn":
        dataset = torchvision.datasets.SVHN(data_root, split="test", download=True, transform=t_rgb)
    elif name == "Caltech101":
        dataset = torchvision.datasets.Caltech101(data_root, download=True, transform=t_resize_224)
    elif name == "Caltech256":
        dataset = torchvision.datasets.Caltech256(data_root, download=True, transform=t_resize_224)
    elif name == "Food101":
        dataset = torchvision.datasets.Food101(data_root, split="test", download=True, transform=t_resize_224)
    elif name == "Flowers102":
        dataset = torchvision.datasets.Flowers102(data_root, split="test", download=True, transform=t_resize_224)
    elif name == "StanfordCars":
        dataset = torchvision.datasets.StanfordCars(data_root, split="test", download=True, transform=t_resize_224)
    elif name == "OxfordPets":
        dataset = torchvision.datasets.OxfordIIITPet(data_root, split="test", download=True, transform=t_resize_224)
    elif name == "Eurosat":
        dataset = torchvision.datasets.EuroSAT(data_root, download=True, transform=t_rgb)
    elif name == "SpeechCommands":
        dataset = torchvision.datasets.SPEECHCOMMANDS(data_root, subset="testing", download=True)
    
    # ═══════════════════════════════════════════════════════════════
    # HUGGINGFACE DATASETS
    # ═══════════════════════════════════════════════════════════════
    
    elif name in [
        # Text classification
        "Imdb", "Sst2", "Sst5", "YelpReviews", "AmazonPolarity", "RottenTomatoes",
        "FinancialSentiment", "TweetSentiment", "AgNews", "Dbpedia", "YahooAnswers",
        "TwentyNewsgroups", "SmsSpam", "HateSpeech", "CivilComments", "Toxicity",
        # Intent
        "ClincIntent", "Banking77", "SnipsIntent",
        # NER
        "Conll2003", "Wnut17",
        # QA
        "Squad", "SquadV2", "TriviaQa", "BoolQ", "CommonsenseQa",
        # Similarity
        "Stsb", "Mrpc", "Qqp", "Snli", "Mnli",
        # Summarization
        "CnnDailymail", "Xsum", "Samsum",
        # Medical
        "PubmedQa", "MedQa",
        # Code
        "CodeSearchNet", "Humaneval", "Mbpp", "Spider",
        # Multilingual
        "Xnli", "AmazonReviewsMulti", "Sberquad",
    ]:
        dataset = load_huggingface_dataset(name, data_root, limit)
    
    # ═══════════════════════════════════════════════════════════════
    # SKLEARN / TABULAR DATASETS
    # ═══════════════════════════════════════════════════════════════
    
    elif name in ["Iris", "Wine", "Diabetes", "BreastCancer", "CaliforniaHousing"]:
        dataset = load_sklearn_dataset(name)
    
    # ═══════════════════════════════════════════════════════════════
    # CUSTOM LOADERS
    # ═══════════════════════════════════════════════════════════════
    
    elif name in ["StanfordDogs", "CatsVsDogs", "ChestXray", "SkinCancer", 
                  "DiabeticRetinopathy", "BrainTumor", "Malaria", "BloodCells", "CovidXray"]:
        dataset = load_image_folder_dataset(name, data_root, t_resize_224)
    
    elif name in ["AdultIncome", "BankMarketing", "CreditDefault", "Titanic", 
                  "HeartDisease", "CreditCardFraud", "NslKdd", "Phishing"]:
        dataset = load_tabular_dataset(name, data_root)
    
    elif name in ["Gtzan", "Esc50", "Urbansound8k", "Nsynth", "Ravdess", 
                  "CremaD", "Iemocap", "Librispeech", "CommonVoice"]:
        dataset = load_audio_dataset(name, data_root)
    
    elif name in ["Electricity", "Weather", "StockPrices", "EcgHeartbeat"]:
        dataset = load_timeseries_dataset(name, data_root)
    
    elif name in ["Cora", "Citeseer", "Qm9"]:
        dataset = load_graph_dataset(name, data_root)
    
    elif name in ["Movielens1m", "Movielens100k"]:
        dataset = load_recommender_dataset(name, data_root)
    
    else:
        raise ValueError(f"Dataset {name} not yet implemented")
    
    if dataset is None:
        raise ValueError(f"Failed to load dataset {name}")
    
    if len(dataset) > limit:
        dataset = torch.utils.data.Subset(dataset, range(limit))
    
    return dataset


def load_huggingface_dataset(name: str, data_root: Path, limit: int):
    """Load dataset from HuggingFace"""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    # Mapping to HuggingFace names
    hf_mapping = {
        "Imdb": ("imdb", "test"),
        "Sst2": ("glue", "sst2", "validation"),
        "AgNews": ("ag_news", "test"),
        "Dbpedia": ("dbpedia_14", "test"),
        "YelpReviews": ("yelp_review_full", "test"),
        "AmazonPolarity": ("amazon_polarity", "test"),
        "RottenTomatoes": ("rotten_tomatoes", "test"),
        "TwentyNewsgroups": ("newsgroup", "test"),
        "SmsSpam": ("sms_spam", "train"),  # no test split
        "Banking77": ("banking77", "test"),
        "ClincIntent": ("clinc_oos", "test"),
        "Conll2003": ("conll2003", "test"),
        "Wnut17": ("wnut_17", "test"),
        "Squad": ("squad", "validation"),
        "SquadV2": ("squad_v2", "validation"),
        "BoolQ": ("boolq", "validation"),
        "CommonsenseQa": ("commonsense_qa", "validation"),
        "Stsb": ("glue", "stsb", "validation"),
        "Mrpc": ("glue", "mrpc", "validation"),
        "Qqp": ("glue", "qqp", "validation"),
        "Snli": ("snli", "test"),
        "Mnli": ("glue", "mnli", "validation_matched"),
        "CnnDailymail": ("cnn_dailymail", "3.0.0", "test"),
        "Xsum": ("xsum", "test"),
        "Samsum": ("samsum", "test"),
        "Xnli": ("xnli", "test"),
        "CodeSearchNet": ("code_search_net", "python", "test"),
        "Humaneval": ("openai_humaneval", "test"),
        "Sberquad": ("sberquad", "test"),
    }
    
    if name not in hf_mapping:
        raise ValueError(f"HuggingFace mapping not found for {name}")
    
    mapping = hf_mapping[name]
    
    if len(mapping) == 2:
        ds = hf_load(mapping[0], split=mapping[1], cache_dir=str(data_root))
    elif len(mapping) == 3:
        ds = hf_load(mapping[0], mapping[1], split=mapping[2], cache_dir=str(data_root))
    else:
        raise ValueError(f"Invalid mapping for {name}")
    
    # Convert to torch dataset
    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, limit):
            self.data = list(hf_ds)[:limit]
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            # Get text and label
            text = item.get("text") or item.get("sentence") or item.get("question") or str(item)
            label = item.get("label", 0)
            # Return as tensor placeholder (text models need tokenization)
            return torch.zeros(768), label  # Placeholder embedding
    
    return HFDataset(ds, limit)


def load_sklearn_dataset(name: str):
    """Load sklearn dataset"""
    from sklearn import datasets as sk_datasets
    from sklearn.model_selection import train_test_split
    
    loaders = {
        "Iris": sk_datasets.load_iris,
        "Wine": sk_datasets.load_wine,
        "Diabetes": sk_datasets.load_diabetes,
        "BreastCancer": sk_datasets.load_breast_cancer,
        "CaliforniaHousing": sk_datasets.fetch_california_housing,
    }
    
    data = loaders[name]()
    X, y = data.data, data.target
    
    # For regression, convert to classification
    if name in ["Diabetes", "CaliforniaHousing"]:
        y = (y > np.median(y)).astype(int)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    class SklearnDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return SklearnDataset(X_test, y_test)


def load_image_folder_dataset(name: str, data_root: Path, transform):
    """Load image dataset from folder or download"""
    console.print(f"[yellow]⚠ {name}: requires manual download to {data_root / name}[/yellow]")
    
    folder = data_root / name / "test"
    if folder.exists():
        import torchvision
        return torchvision.datasets.ImageFolder(str(folder), transform=transform)
    
    raise ValueError(f"Dataset {name} not found. Please download to {folder}")


def load_tabular_dataset(name: str, data_root: Path):
    """Load tabular dataset"""
    console.print(f"[yellow]⚠ {name}: tabular dataset - using placeholder[/yellow]")
    
    # Placeholder - return random data
    class TabularDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=1000, n_features=20, n_classes=2):
            self.X = torch.randn(n_samples, n_features)
            self.y = torch.randint(0, n_classes, (n_samples,))
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return TabularDataset()


def load_audio_dataset(name: str, data_root: Path):
    """Load audio dataset"""
    console.print(f"[yellow]⚠ {name}: audio dataset - using placeholder[/yellow]")
    
    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=1000, n_features=128):
            self.X = torch.randn(n_samples, n_features)
            self.y = torch.randint(0, 10, (n_samples,))
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return AudioDataset()


def load_timeseries_dataset(name: str, data_root: Path):
    """Load time series dataset"""
    console.print(f"[yellow]⚠ {name}: timeseries dataset - using placeholder[/yellow]")
    
    class TimeSeriesDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=1000, seq_len=100, n_features=1):
            self.X = torch.randn(n_samples, seq_len, n_features)
            self.y = torch.randint(0, 2, (n_samples,))
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return TimeSeriesDataset()


def load_graph_dataset(name: str, data_root: Path):
    """Load graph dataset"""
    console.print(f"[yellow]⚠ {name}: graph dataset - using placeholder[/yellow]")
    
    class GraphDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=1000, n_features=128):
            self.X = torch.randn(n_samples, n_features)
            self.y = torch.randint(0, 7, (n_samples,))
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return GraphDataset()


def load_recommender_dataset(name: str, data_root: Path):
    """Load recommender dataset"""
    console.print(f"[yellow]⚠ {name}: recommender dataset - using placeholder[/yellow]")
    
    class RecommenderDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples=1000):
            self.X = torch.randn(n_samples, 64)
            self.y = torch.randint(0, 5, (n_samples,))
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    return RecommenderDataset()


def compute_embeddings(
    encoder: nn.Module,
    dataset,
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute embeddings"""
    encoder = encoder.to(device)
    encoder.eval()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_emb = []
    all_labels = []
    
    with torch.no_grad():
        with Progress() as progress:
            task = progress.add_task("Computing embeddings...", total=len(loader))
            
            for batch in loader:
                inputs, labels = batch[0].to(device), batch[1]
                
                emb = encoder(inputs)
                
                # Flatten if needed
                if len(emb.shape) > 2:
                    emb = emb.view(emb.size(0), -1)
                
                all_emb.append(emb.cpu().numpy())
                all_labels.append(labels.numpy())
                
                progress.advance(task)
    
    return np.concatenate(all_emb), np.concatenate(all_labels)


def load_model(path: str) -> nn.Module:
    """Load PyTorch model"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, nn.Module):
        return checkpoint
    
    if isinstance(checkpoint, dict):
        # Try common keys
        for key in ["model", "net", "network", "state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], nn.Module):
                return checkpoint[key]
        
        # state_dict only - can't use
        if "state_dict" in checkpoint or all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            raise ValueError("File contains only state_dict, need full model with architecture")
    
    raise ValueError(f"Cannot load model from {path}")


def create_package(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    limit: int = 10000,
    batch_size: int = 64,
    device: str = "cpu",
) -> bool:
    """Create Decloud package"""
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # 1. Load model
    console.print(f"\n[cyan]Loading model...[/cyan]")
    try:
        model = load_model(model_path)
        model.eval()
        console.print(f"[green]✓ Loaded: {type(model).__name__}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False
    
    # 2. Extract layers
    console.print(f"\n[cyan]Analyzing architecture...[/cyan]")
    layers = extract_layers(model)
    
    if not layers:
        console.print("[red]✗ No layers found[/red]")
        return False
    
    total_params = sum(l.num_params for l in layers)
    console.print(f"[green]✓ Found {len(layers)} layers, {total_params:,} params[/green]")
    
    # 3. Find split point (15% params = head)
    split_idx = find_split_index(layers, HEAD_PARAMS_RATIO)
    display_split(layers, split_idx)
    
    encoder_layers = layers[:split_idx]
    head_layers = layers[split_idx:]
    
    if not head_layers:
        console.print("[red]✗ No head layers[/red]")
        return False
    
    # Check for unknown layers in head
    unknown = [l for l in head_layers if l.type_name.startswith("Unknown")]
    if unknown:
        console.print(f"\n[yellow]⚠ Unknown layers in head:[/yellow]")
        for l in unknown:
            console.print(f"  {l.name}: {l.type_name}")
        if not Confirm.ask("Continue?", default=False):
            return False
    
    # 4. Build encoder
    console.print(f"\n[cyan]Building encoder...[/cyan]")
    encoder = build_encoder(model, layers, split_idx)
    console.print(f"[green]✓ Encoder ready[/green]")
    
    # 5. Load dataset & compute embeddings
    console.print(f"\n[cyan]Loading {dataset_name}...[/cyan]")
    try:
        dataset = load_dataset(dataset_name, limit)
        console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False
    
    console.print(f"\n[cyan]Computing embeddings ({device})...[/cyan]")
    try:
        embeddings, labels = compute_embeddings(encoder, dataset, batch_size, device)
        console.print(f"[green]✓ Embeddings: {embeddings.shape}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Build head config & weights
    console.print(f"\n[cyan]Extracting head...[/cyan]")
    head_config = build_head_config(head_layers)
    head_config["metadata"] = {
        "dataset": dataset_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_samples": int(embeddings.shape[0]),
        "head_params_ratio": HEAD_PARAMS_RATIO,
    }
    
    head_weights = extract_head_weights(head_layers)
    console.print(f"[green]✓ Head: {len(head_config['layers'])} layers, {len(head_weights)} weight tensors[/green]")
    
    # 7. Save package
    console.print(f"\n[cyan]Saving to {output}...[/cyan]")
    
    # config.json
    with open(output / "config.json", "w") as f:
        json.dump(head_config, f, indent=2)
    console.print(f"[green]  ✓ config.json[/green]")
    
    # embeddings.safetensors
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    save_safetensors({"embeddings": emb_tensor}, str(output / "embeddings.safetensors"))
    emb_size_mb = embeddings.nbytes / 1024 / 1024
    console.print(f"[green]  ✓ embeddings.safetensors ({emb_size_mb:.1f} MB)[/green]")
    
    # head.safetensors
    if head_weights:
        save_safetensors(head_weights, str(output / "head.safetensors"))
        console.print(f"[green]  ✓ head.safetensors[/green]")
    
    # Done
    console.print(f"\n{'═'*50}")
    console.print(f"[bold green]✓ Package created![/bold green]")
    console.print(f"\n[dim]Files:[/dim]")
    for f in output.iterdir():
        size = f.stat().st_size / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        console.print(f"  {f.name}: {size:.1f} {unit}")
    
    console.print(f"\n[cyan]Next: upload to IPFS and create round[/cyan]")
    return True


def interactive():
    """Interactive mode"""
    console.print("\n[bold cyan]═══ Decloud Creator Kit ═══[/bold cyan]")
    console.print(f"[dim]Head = last {HEAD_PARAMS_RATIO*100:.0f}% of parameters[/dim]\n")
    
    # Model
    model_path = Prompt.ask("Model path (.pt / .pth)")
    if not Path(model_path).exists():
        console.print(f"[red]File not found: {model_path}[/red]")
        return
    
    # Dataset
    console.print(f"\n[dim]Available: {', '.join(DATASETS)}[/dim]")
    dataset = Prompt.ask("Dataset", default="Cifar10")
    
    # Output
    output = Prompt.ask("Output directory", default="./decloud_package")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Device: {device}[/dim]")
    
    create_package(model_path, dataset, output, device=device)


def main():
    parser = argparse.ArgumentParser(description="Decloud Creator Kit")
    parser.add_argument("--model", "-m", help="Model path")
    parser.add_argument("--dataset", "-d", help="Dataset name")
    parser.add_argument("--output", "-o", default="./decloud_package", help="Output dir")
    parser.add_argument("--limit", "-l", type=int, default=10000, help="Max samples")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    if args.model and args.dataset:
        create_package(
            args.model, args.dataset, args.output,
            args.limit, args.batch_size, args.device
        )
    else:
        interactive()


if __name__ == "__main__":
    main()

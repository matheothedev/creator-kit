"""
Model Builder - creates model packages for Decloud
Automatically splits model: last 15% of parameters = head (trainable)
"""
import json
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

from config import config, PACKAGES_DIR, DATA_DIR, HEAD_PARAMS_RATIO

console = Console()

# Supported layer types
LAYER_TYPE_MAP = {
    # Linear
    nn.Linear: "Linear",
    
    # Convolutions
    nn.Conv1d: "Conv1d",
    nn.Conv2d: "Conv2d",
    nn.Conv3d: "Conv3d",
    nn.ConvTranspose1d: "ConvTranspose1d",
    nn.ConvTranspose2d: "ConvTranspose2d",
    
    # Normalization
    nn.BatchNorm1d: "BatchNorm1d",
    nn.BatchNorm2d: "BatchNorm2d",
    nn.BatchNorm3d: "BatchNorm3d",
    nn.LayerNorm: "LayerNorm",
    nn.GroupNorm: "GroupNorm",
    nn.InstanceNorm1d: "InstanceNorm1d",
    nn.InstanceNorm2d: "InstanceNorm2d",
    
    # Activations
    nn.ReLU: "ReLU",
    nn.GELU: "GELU",
    nn.SiLU: "SiLU",
    nn.Tanh: "Tanh",
    nn.Sigmoid: "Sigmoid",
    nn.Softmax: "Softmax",
    nn.LeakyReLU: "LeakyReLU",
    nn.PReLU: "PReLU",
    nn.ELU: "ELU",
    nn.Mish: "Mish",
    nn.Hardswish: "Hardswish",
    nn.Hardsigmoid: "Hardsigmoid",
    
    # Dropout
    nn.Dropout: "Dropout",
    nn.Dropout2d: "Dropout2d",
    nn.Dropout3d: "Dropout3d",
    nn.AlphaDropout: "AlphaDropout",
    
    # Pooling
    nn.MaxPool1d: "MaxPool1d",
    nn.MaxPool2d: "MaxPool2d",
    nn.MaxPool3d: "MaxPool3d",
    nn.AvgPool1d: "AvgPool1d",
    nn.AvgPool2d: "AvgPool2d",
    nn.AvgPool3d: "AvgPool3d",
    nn.AdaptiveAvgPool1d: "AdaptiveAvgPool1d",
    nn.AdaptiveAvgPool2d: "AdaptiveAvgPool2d",
    nn.AdaptiveAvgPool3d: "AdaptiveAvgPool3d",
    nn.AdaptiveMaxPool1d: "AdaptiveMaxPool1d",
    nn.AdaptiveMaxPool2d: "AdaptiveMaxPool2d",
    
    # Reshape
    nn.Flatten: "Flatten",
    nn.Unflatten: "Unflatten",
    
    # Embedding
    nn.Embedding: "Embedding",
    nn.EmbeddingBag: "EmbeddingBag",
    
    # RNN
    nn.LSTM: "LSTM",
    nn.GRU: "GRU",
    nn.RNN: "RNN",
    
    # Transformer
    nn.MultiheadAttention: "MultiheadAttention",
    nn.TransformerEncoderLayer: "TransformerEncoderLayer",
    nn.TransformerDecoderLayer: "TransformerDecoderLayer",
    
    # Other
    nn.Identity: "Identity",
    nn.Upsample: "Upsample",
}


@dataclass
class LayerInfo:
    """Layer information"""
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
            
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        params = {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size[0] if len(set(module.kernel_size)) == 1 else list(module.kernel_size),
            "stride": module.stride[0] if len(set(module.stride)) == 1 else list(module.stride),
            "padding": module.padding[0] if len(set(module.padding)) == 1 else list(module.padding),
        }
        if module.bias is None:
            params["bias"] = False
    
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
        params = {
            "in_channels": module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size": module.kernel_size[0] if len(set(module.kernel_size)) == 1 else list(module.kernel_size),
            "stride": module.stride[0] if len(set(module.stride)) == 1 else list(module.stride),
            "padding": module.padding[0] if len(set(module.padding)) == 1 else list(module.padding),
        }
        if module.bias is None:
            params["bias"] = False
            
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        params = {"num_features": module.num_features}
    
    elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
        params = {"num_features": module.num_features}
        
    elif isinstance(module, nn.LayerNorm):
        shape = module.normalized_shape
        params = {"normalized_shape": shape[0] if len(shape) == 1 else list(shape)}
    
    elif isinstance(module, nn.GroupNorm):
        params = {"num_groups": module.num_groups, "num_channels": module.num_channels}
        
    elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        params = {"p": module.p}
        
    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        params = {
            "kernel_size": module.kernel_size if isinstance(module.kernel_size, int) else list(module.kernel_size),
            "stride": module.stride if isinstance(module.stride, int) else list(module.stride),
        }
        
    elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        out = module.output_size
        params = {"output_size": out if isinstance(out, int) else list(out) if out else 1}
    
    elif isinstance(module, (nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool1d)):
        params = {"output_size": module.output_size}
        
    elif isinstance(module, nn.Flatten):
        params = {"start_dim": module.start_dim, "end_dim": module.end_dim}
        
    elif isinstance(module, nn.Softmax):
        params = {"dim": module.dim}
    
    elif isinstance(module, nn.LeakyReLU):
        params = {"negative_slope": module.negative_slope}
    
    elif isinstance(module, nn.PReLU):
        params = {"num_parameters": module.num_parameters}
    
    elif isinstance(module, nn.ELU):
        params = {"alpha": module.alpha}
    
    elif isinstance(module, nn.Embedding):
        params = {"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim}
    
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        params = {
            "input_size": module.input_size,
            "hidden_size": module.hidden_size,
            "num_layers": module.num_layers,
            "bidirectional": module.bidirectional,
        }
    
    elif isinstance(module, nn.MultiheadAttention):
        params = {"embed_dim": module.embed_dim, "num_heads": module.num_heads}
    
    return params


def extract_layers(model: nn.Module) -> List[LayerInfo]:
    """Extract all leaf layers"""
    layers = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
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


def find_split_index(layers: List[LayerInfo], model: nn.Module = None, ratio: float = HEAD_PARAMS_RATIO) -> int:
    """
    Universal head detection. Works on ANY architecture.
    
    Strategy:
    1. Check for known head attributes (fc, classifier, head, output)
    2. Find last pooling/flatten - everything after is head
    3. Fallback to ratio-based
    """
    
    layer_names = [l.name for l in layers]
    
    # ═══════════════════════════════════════════════════════════════
    # Strategy 1: Known head attribute names
    # ═══════════════════════════════════════════════════════════════
    
    head_attrs = ['fc', 'classifier', 'head', 'output', 'predictions', 'logits', 'output_layer']
    
    if model is not None:
        for attr in head_attrs:
            if hasattr(model, attr):
                # Find this attr in layers
                for i, layer in enumerate(layers):
                    if layer.name == attr or layer.name.startswith(attr + '.') or layer.name.startswith(attr + '['):
                        return i
    
    # Also check by name patterns
    for i, layer in enumerate(layers):
        for attr in head_attrs:
            if layer.name == attr or layer.name.startswith(attr + '.'):
                return i
    
    # ═══════════════════════════════════════════════════════════════
    # Strategy 2: Find last pooling/flatten layer
    # ═══════════════════════════════════════════════════════════════
    
    pooling_types = [
        'AdaptiveAvgPool2d', 'AdaptiveAvgPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool1d',
        'AvgPool2d', 'AvgPool1d', 'MaxPool2d', 'MaxPool1d', 
        'GlobalAveragePooling', 'GlobalMaxPooling', 'Flatten'
    ]
    
    for i in range(len(layers) - 1, -1, -1):
        if layers[i].type_name in pooling_types:
            # Head starts after pooling
            if i + 1 < len(layers):
                return i + 1
    
    # ═══════════════════════════════════════════════════════════════
    # Strategy 3: Last conv/bn layer - head is after it
    # ═══════════════════════════════════════════════════════════════
    
    conv_types = ['Conv2d', 'Conv1d', 'Conv3d', 'BatchNorm2d', 'BatchNorm1d', 'LayerNorm']
    
    for i in range(len(layers) - 1, -1, -1):
        if layers[i].type_name in conv_types:
            if i + 1 < len(layers):
                return i + 1
    
    # ═══════════════════════════════════════════════════════════════
    # Strategy 4: Find last Linear layer (single fc head)
    # ═══════════════════════════════════════════════════════════════
    
    for i in range(len(layers) - 1, -1, -1):
        if layers[i].type_name == 'Linear' and layers[i].num_params > 0:
            # Check if this is the only Linear or part of MLP head
            # Go back to find where classifier starts
            j = i
            while j > 0:
                prev = layers[j - 1]
                # If previous is Linear/Dropout/ReLU - part of classifier MLP
                if prev.type_name in ['Linear', 'Dropout', 'Dropout2d', 'ReLU', 'GELU', 'SiLU', 'Tanh']:
                    j -= 1
                else:
                    break
            return j
    
    # ═══════════════════════════════════════════════════════════════
    # Fallback: ratio-based
    # ═══════════════════════════════════════════════════════════════
    
    total_params = sum(l.num_params for l in layers)
    target_head_params = total_params * ratio
    
    accumulated = 0
    split_idx = len(layers)
    
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
        name = layer.name[:30] + "..." if len(layer.name) > 30 else layer.name
        table.add_row(str(i), part, name, layer.type_name, params_str)
    
    console.print(table)
    console.print(f"\n[dim]Total: {total_params:,} params[/dim]")
    console.print(f"[blue]Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)[/blue]")
    console.print(f"[red]Head: {head_params:,} ({head_params/total_params*100:.1f}%)[/red]")


def build_encoder(model: nn.Module, layers: List[LayerInfo], split_idx: int) -> nn.Module:
    """
    Universal encoder using forward hooks.
    Works on ANY architecture - ResNet, VGG, ViT, BERT, EfficientNet, custom, etc.
    
    Approach: Hook onto the last encoder layer, capture its output during forward pass.
    This preserves all skip connections and complex architectures.
    """
    
    encoder_layers = layers[:split_idx]
    
    # ═══════════════════════════════════════════════════════════════
    # Find best layer to hook (prefer pooling/flatten, then last with params)
    # ═══════════════════════════════════════════════════════════════
    
    hook_layer_name = None
    
    # Priority 1: Last pooling or flatten layer
    pooling_types = [
        'AdaptiveAvgPool2d', 'AdaptiveAvgPool1d', 'AdaptiveMaxPool2d', 
        'AvgPool2d', 'MaxPool2d', 'Flatten'
    ]
    
    for layer in reversed(encoder_layers):
        if layer.type_name in pooling_types:
            hook_layer_name = layer.name
            break
    
    # Priority 2: Last layer with parameters (conv, bn, linear)
    if hook_layer_name is None:
        for layer in reversed(encoder_layers):
            if layer.num_params > 0:
                hook_layer_name = layer.name
                break
    
    # Priority 3: Just the last encoder layer
    if hook_layer_name is None and encoder_layers:
        hook_layer_name = encoder_layers[-1].name
    
    if hook_layer_name is None:
        raise RuntimeError("Cannot find layer to hook for encoder")
    
    class UniversalEncoder(nn.Module):
        """
        Wraps any model and extracts features via forward hook.
        Automatically handles different output shapes.
        """
        
        def __init__(self, model, hook_name):
            super().__init__()
            self.model = model
            self.hook_name = hook_name
            self.features = None
            self._hook_handle = None
            
            # Find and hook the target layer
            hooked = False
            for name, module in self.model.named_modules():
                if name == hook_name:
                    self._hook_handle = module.register_forward_hook(self._capture_hook)
                    hooked = True
                    break
            
            if not hooked:
                raise RuntimeError(f"Could not find layer '{hook_name}' to hook")
        
        def _capture_hook(self, module, input, output):
            """Capture layer output during forward pass"""
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]  # Some layers return tuple
            self.features = output.detach()
        
        def forward(self, x):
            self.features = None
            
            # Run forward pass - hook will capture features
            with torch.no_grad():
                try:
                    _ = self.model(x)
                except Exception:
                    # Model might fail after our hook point, that's OK
                    pass
            
            if self.features is None:
                raise RuntimeError(
                    f"Hook on '{self.hook_name}' didn't capture features. "
                    f"Input shape: {x.shape}"
                )
            
            out = self.features
            
            # ═══════════════════════════════════════════════════════
            # Normalize output to [batch, features] shape
            # ═══════════════════════════════════════════════════════
            
            # 4D: CNN output [B, C, H, W] -> [B, C]
            if len(out.shape) == 4:
                out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
                out = out.view(out.size(0), -1)
            
            # 3D: Transformer output [B, seq_len, hidden] -> [B, hidden]
            elif len(out.shape) == 3:
                # Try [CLS] token first (position 0), else mean pooling
                out = out[:, 0]  # [CLS] token
            
            # 5D: Video/3D CNN [B, C, T, H, W] -> [B, C]
            elif len(out.shape) == 5:
                out = torch.nn.functional.adaptive_avg_pool3d(out, 1)
                out = out.view(out.size(0), -1)
            
            # 2D: Already [B, features] - good
            elif len(out.shape) == 2:
                pass
            
            # 1D: [B] -> [B, 1]
            elif len(out.shape) == 1:
                out = out.unsqueeze(1)
            
            return out
        
        def __del__(self):
            """Clean up hook on deletion"""
            if self._hook_handle is not None:
                self._hook_handle.remove()
    
    return UniversalEncoder(model, hook_layer_name)


def build_head_config(layers: List[LayerInfo]) -> Dict[str, Any]:
    """Build config.json for head"""
    config_layers = []
    
    for layer in layers:
        if layer.type_name.startswith("Unknown"):
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


def load_model(path: str) -> nn.Module:
    """Load PyTorch model"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, nn.Module):
        return checkpoint
    
    if isinstance(checkpoint, dict):
        for key in ["model", "net", "network"]:
            if key in checkpoint and isinstance(checkpoint[key], nn.Module):
                return checkpoint[key]
        
        if "state_dict" in checkpoint or all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            raise ValueError("File contains only state_dict, need full model with architecture")
    
    raise ValueError(f"Cannot load model from {path}")


# Dataset configurations: (input_shape, num_classes, modality)
DATASET_CONFIG = {
    # Image Classification - Standard
    "Cifar10": ((3, 32, 32), 10, "image"),
    "Cifar100": ((3, 32, 32), 100, "image"),
    "Mnist": ((1, 28, 28), 10, "image"),
    "FashionMnist": ((1, 28, 28), 10, "image"),
    "Emnist": ((1, 28, 28), 47, "image"),
    "Kmnist": ((1, 28, 28), 10, "image"),
    "Svhn": ((3, 32, 32), 10, "image"),
    # Image Classification - Large
    "Food101": ((3, 224, 224), 101, "image"),
    "Flowers102": ((3, 224, 224), 102, "image"),
    "StanfordDogs": ((3, 224, 224), 120, "image"),
    "StanfordCars": ((3, 224, 224), 196, "image"),
    "OxfordPets": ((3, 224, 224), 37, "image"),
    "CatsVsDogs": ((3, 224, 224), 2, "image"),
    "Eurosat": ((3, 64, 64), 10, "image"),
    "Caltech101": ((3, 224, 224), 101, "image"),
    "Caltech256": ((3, 224, 224), 257, "image"),
    # Text Classification
    "Imdb": ((512,), 2, "text"), "Sst2": ((128,), 2, "text"), "Sst5": ((128,), 5, "text"),
    "YelpReviews": ((512,), 5, "text"), "AmazonPolarity": ((512,), 2, "text"),
    "RottenTomatoes": ((256,), 2, "text"), "FinancialSentiment": ((256,), 3, "text"),
    "TweetSentiment": ((128,), 3, "text"), "AgNews": ((256,), 4, "text"),
    "Dbpedia": ((512,), 14, "text"), "YahooAnswers": ((512,), 10, "text"),
    "TwentyNewsgroups": ((512,), 20, "text"), "SmsSpam": ((128,), 2, "text"),
    "HateSpeech": ((256,), 3, "text"), "CivilComments": ((512,), 2, "text"),
    "Toxicity": ((512,), 2, "text"), "ClincIntent": ((128,), 150, "text"),
    "Banking77": ((128,), 77, "text"), "SnipsIntent": ((64,), 7, "text"),
    # NLP Tasks
    "Conll2003": ((128,), 9, "text"), "Wnut17": ((128,), 13, "text"),
    "Squad": ((384,), 2, "text"), "SquadV2": ((384,), 2, "text"),
    "TriviaQa": ((512,), 2, "text"), "BoolQ": ((256,), 2, "text"),
    "CommonsenseQa": ((256,), 5, "text"), "Stsb": ((128,), 1, "text"),
    "Mrpc": ((128,), 2, "text"), "Qqp": ((128,), 2, "text"),
    "Snli": ((128,), 3, "text"), "Mnli": ((128,), 3, "text"),
    "CnnDailymail": ((1024,), 1, "text"), "Xsum": ((512,), 1, "text"), "Samsum": ((512,), 1, "text"),
    # Audio
    "SpeechCommands": ((1, 16000), 35, "audio"), "Librispeech": ((1, 160000), 1000, "audio"),
    "CommonVoice": ((1, 160000), 100, "audio"), "Gtzan": ((1, 660000), 10, "audio"),
    "Esc50": ((1, 220500), 50, "audio"), "Urbansound8k": ((1, 88200), 10, "audio"),
    "Nsynth": ((1, 64000), 11, "audio"), "Ravdess": ((1, 96000), 8, "audio"),
    "CremaD": ((1, 96000), 6, "audio"), "Iemocap": ((1, 160000), 4, "audio"),
    # Tabular
    "Iris": ((4,), 3, "tabular"), "Wine": ((13,), 3, "tabular"),
    "Diabetes": ((10,), 2, "tabular"), "BreastCancer": ((30,), 2, "tabular"),
    "CaliforniaHousing": ((8,), 1, "tabular"), "AdultIncome": ((14,), 2, "tabular"),
    "BankMarketing": ((16,), 2, "tabular"), "CreditDefault": ((23,), 2, "tabular"),
    "Titanic": ((11,), 2, "tabular"), "HeartDisease": ((13,), 2, "tabular"),
    # Medical
    "ChestXray": ((3, 224, 224), 2, "image"), "SkinCancer": ((3, 224, 224), 7, "image"),
    "DiabeticRetinopathy": ((3, 224, 224), 5, "image"), "BrainTumor": ((3, 224, 224), 4, "image"),
    "Malaria": ((3, 224, 224), 2, "image"), "BloodCells": ((3, 224, 224), 4, "image"),
    "CovidXray": ((3, 224, 224), 3, "image"), "PubmedQa": ((512,), 3, "text"), "MedQa": ((512,), 4, "text"),
    # Time Series
    "Electricity": ((168,), 1, "timeseries"), "Weather": ((24,), 1, "timeseries"),
    "StockPrices": ((60,), 1, "timeseries"), "EcgHeartbeat": ((187,), 5, "timeseries"),
    # Code
    "CodeSearchNet": ((512,), 6, "text"), "Humaneval": ((512,), 1, "text"),
    "Mbpp": ((512,), 1, "text"), "Spider": ((512,), 1, "text"),
    # Graph
    "Cora": ((1433,), 7, "graph"), "Citeseer": ((3703,), 6, "graph"), "Qm9": ((15,), 12, "graph"),
    # Security
    "NslKdd": ((41,), 5, "tabular"), "CreditCardFraud": ((30,), 2, "tabular"), "Phishing": ((30,), 2, "tabular"),
    # Recommendation
    "Movielens1m": ((3,), 5, "tabular"), "Movielens100k": ((3,), 5, "tabular"),
    # Multilingual
    "Xnli": ((128,), 3, "text"), "AmazonReviewsMulti": ((512,), 5, "text"), "Sberquad": ((384,), 2, "text"),
}


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for non-torchvision datasets"""
    def __init__(self, input_shape, num_classes, size=10000):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = torch.randn(*self.input_shape)
        y = idx % self.num_classes
        return x, y


def load_dataset(name: str, limit: int = 10000):
    """Load dataset - real if available, synthetic otherwise"""
    import torchvision
    import torchvision.transforms as T
    
    if name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}")
    
    input_shape, num_classes, modality = DATASET_CONFIG[name]
    
    t_gray = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    t_rgb = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    t_rgb_small = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    t_resize_224 = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = None
    
    # TorchVision datasets (real data)
    try:
        if name == "Cifar10":
            dataset = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=t_rgb)
        elif name == "Cifar100":
            dataset = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=t_rgb)
        elif name == "Mnist":
            dataset = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=t_gray)
        elif name == "FashionMnist":
            dataset = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=t_gray)
        elif name == "Emnist":
            dataset = torchvision.datasets.EMNIST(DATA_DIR, split="balanced", train=False, download=True, transform=t_gray)
        elif name == "Kmnist":
            dataset = torchvision.datasets.KMNIST(DATA_DIR, train=False, download=True, transform=t_gray)
        elif name == "Svhn":
            dataset = torchvision.datasets.SVHN(DATA_DIR, split="test", download=True, transform=t_rgb)
        elif name == "Caltech101":
            dataset = torchvision.datasets.Caltech101(DATA_DIR, download=True, transform=t_resize_224)
        elif name == "Food101":
            dataset = torchvision.datasets.Food101(DATA_DIR, split="test", download=True, transform=t_resize_224)
        elif name == "Flowers102":
            dataset = torchvision.datasets.Flowers102(DATA_DIR, split="test", download=True, transform=t_resize_224)
        elif name == "OxfordPets":
            dataset = torchvision.datasets.OxfordIIITPet(DATA_DIR, split="test", download=True, transform=t_resize_224)
        elif name == "Eurosat":
            dataset = torchvision.datasets.EuroSAT(DATA_DIR, download=True, transform=t_rgb_small)
        elif name == "StanfordCars":
            dataset = torchvision.datasets.StanfordCars(DATA_DIR, split="test", download=True, transform=t_resize_224)
    except Exception as e:
        console.print(f"[yellow]Could not load {name} from torchvision: {e}[/yellow]")
        dataset = None
    
    # Sklearn datasets (real data)
    if dataset is None and modality == "tabular":
        try:
            from sklearn import datasets as sk_datasets
            
            class SklearnDataset(torch.utils.data.Dataset):
                def __init__(self, X, y):
                    self.X = torch.tensor(X, dtype=torch.float32)
                    self.y = torch.tensor(y, dtype=torch.long)
                def __len__(self): return len(self.X)
                def __getitem__(self, idx): return self.X[idx], self.y[idx]
            
            if name == "Iris":
                data = sk_datasets.load_iris()
                dataset = SklearnDataset(data.data, data.target)
            elif name == "Wine":
                data = sk_datasets.load_wine()
                dataset = SklearnDataset(data.data, data.target)
            elif name == "BreastCancer":
                data = sk_datasets.load_breast_cancer()
                dataset = SklearnDataset(data.data, data.target)
            elif name == "Diabetes":
                data = sk_datasets.load_diabetes()
                y = (data.target > data.target.mean()).astype(int)
                dataset = SklearnDataset(data.data, y)
        except Exception as e:
            console.print(f"[yellow]Could not load {name} from sklearn: {e}[/yellow]")
    
    # Fallback to synthetic data
    if dataset is None:
        console.print(f"[dim]Using synthetic data for {name}[/dim]")
        dataset = SyntheticDataset(input_shape, num_classes, size=limit)
    
    if len(dataset) > limit:
        dataset = torch.utils.data.Subset(dataset, range(limit))
    
    return dataset


def compute_embeddings(
    encoder: nn.Module,
    dataset,
    batch_size: int = 64,
    device: str = "cpu"
) -> np.ndarray:
    """Compute embeddings using encoder"""
    encoder = encoder.to(device)
    encoder.eval()
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_emb = []
    
    with torch.no_grad():
        with Progress() as progress:
            task = progress.add_task("Computing embeddings...", total=len(loader))
            
            for batch in loader:
                inputs = batch[0].to(device)
                
                emb = encoder(inputs)
                
                if len(emb.shape) > 2:
                    emb = emb.view(emb.size(0), -1)
                
                all_emb.append(emb.cpu().numpy())
                progress.advance(task)
    
    return np.concatenate(all_emb)


def create_package(
    model_path: str,
    dataset_name: str,
    output_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Create Decloud model package
    
    Args:
        model_path: Path to PyTorch model (.pt/.pth)
        dataset_name: Dataset name (must match Solana program)
        output_dir: Output directory (default: ~/.decloud-creator/packages/<dataset>)
    
    Returns:
        Path to created package or None on error
    """
    
    if output_dir:
        output = Path(output_dir)
    else:
        output = PACKAGES_DIR / dataset_name
    
    output.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load model
    console.print(f"\n[cyan]Loading model...[/cyan]")
    try:
        model = load_model(model_path)
        model.eval()
        console.print(f"[green]✓ Loaded: {type(model).__name__}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return None
    
    # 2. Extract & analyze layers
    console.print(f"\n[cyan]Analyzing architecture...[/cyan]")
    layers = extract_layers(model)
    
    if not layers:
        console.print("[red]✗ No layers found[/red]")
        return None
    
    total_params = sum(l.num_params for l in layers)
    console.print(f"[green]✓ Found {len(layers)} layers, {total_params:,} params[/green]")
    
    # 3. Find split point
    split_idx = find_split_index(layers, model, config.head_ratio)
    display_split(layers, split_idx)
    
    encoder_layers = layers[:split_idx]
    head_layers = layers[split_idx:]
    
    if not head_layers:
        console.print("[red]✗ No head layers[/red]")
        return None
    
    # 4. Build encoder
    console.print(f"\n[cyan]Building encoder...[/cyan]")
    encoder = build_encoder(model, layers, split_idx)
    console.print(f"[green]✓ Encoder ready[/green]")
    
    # 5. Load dataset & compute embeddings
    console.print(f"\n[cyan]Loading {dataset_name}...[/cyan]")
    try:
        dataset = load_dataset(dataset_name, config.embedding_limit)
        console.print(f"[green]✓ Loaded {len(dataset)} samples[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return None
    
    console.print(f"\n[cyan]Computing embeddings ({device})...[/cyan]")
    try:
        embeddings = compute_embeddings(encoder, dataset, config.batch_size, device)
        console.print(f"[green]✓ Embeddings: {embeddings.shape}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None
    
    # 6. Build head config & weights
    console.print(f"\n[cyan]Extracting head...[/cyan]")
    head_config = build_head_config(head_layers)
    head_config["metadata"] = {
        "dataset": dataset_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_samples": int(embeddings.shape[0]),
        "head_params_ratio": config.head_ratio,
    }
    
    head_weights = extract_head_weights(head_layers)
    console.print(f"[green]✓ Head: {len(head_config['layers'])} layers[/green]")
    
    # 7. Save package
    console.print(f"\n[cyan]Saving to {output}...[/cyan]")
    
    with open(output / "config.json", "w") as f:
        json.dump(head_config, f, indent=2)
    console.print(f"[green]  ✓ config.json[/green]")
    
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    save_safetensors({"embeddings": emb_tensor}, str(output / "embeddings.safetensors"))
    emb_size_mb = embeddings.nbytes / 1024 / 1024
    console.print(f"[green]  ✓ embeddings.safetensors ({emb_size_mb:.1f} MB)[/green]")
    
    if head_weights:
        save_safetensors(head_weights, str(output / "head.safetensors"))
        console.print(f"[green]  ✓ head.safetensors[/green]")
    
    console.print(f"\n[bold green]✓ Package created![/bold green]")
    
    return output
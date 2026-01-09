# Decloud Creator Kit

Create and manage federated learning rounds on Solana.

## Installation

```bash
pip install -e .
```

## Setup

```bash
decloud-creator setup
```

Enter:
- Solana private key (base58)
- Network (devnet/mainnet)
- Pinata JWT for IPFS uploads

## Commands

### Full Workflow (Recommended)

```bash
# Build, upload, and create round in one command
decloud-creator launch -m model.pt -d Cifar10 -r 0.5
```

### Step by Step

```bash
# 1. Build package from model
decloud-creator build -m model.pt -d Cifar10

# 2. Upload to IPFS
decloud-creator upload -p ~/.decloud-creator/packages/Cifar10 -d Cifar10

# 3. Create round
decloud-creator create -c <CID> -d Cifar10 -r 0.5
```

### Round Management

```bash
# View your rounds
decloud-creator my-rounds

# Round details
decloud-creator info <round_id>

# Finalize (after trainers submitted)
decloud-creator finalize <round_id>

# Force finalize (after 12h deadline)
decloud-creator force-finalize <round_id>

# Cancel (only if no participants)
decloud-creator cancel <round_id>

# Withdraw remainder after finalize
decloud-creator withdraw <round_id>
```

### Downloads

```bash
# Download gradient from trainer
decloud-creator download-gradient <CID>

# Download base model
decloud-creator download-model <CID>
```

### Info

```bash
# Status
decloud-creator status

# Balance
decloud-creator balance

# Available datasets
decloud-creator datasets
```

## How It Works

1. **Build Package**: Takes your PyTorch model, splits it into encoder (frozen) + head (trainable)
   - Head = last 15% of parameters
   - Computes embeddings on test dataset

2. **Upload to IPFS**: Uploads via Pinata
   - `config.json` - head architecture
   - `head.safetensors` - head weights
   - `embeddings.safetensors` - test embeddings

3. **Create Round**: Locks reward in Solana smart contract

4. **Training Flow**:
   - Validators prevalidate (check base accuracy)
   - Trainers train head on their data, submit gradients
   - Validators postvalidate (check improved accuracy)
   - Creator finalizes round
   - Rewards distributed based on improvement

## Package Structure

```
package/
├── config.json           # Head architecture
├── head.safetensors      # Head weights  
└── embeddings.safetensors # Test embeddings
```

## Supported Datasets

**Image**: Cifar10, Cifar100, Mnist, FashionMnist, Emnist, Kmnist, Food101, Flowers102, Svhn, Caltech101, Eurosat

**Text**: Imdb, Sst2, AgNews, Dbpedia, YelpReviews, AmazonPolarity

**Tabular**: Iris, Wine, Diabetes, BreastCancer, CaliforniaHousing

**Medical**: ChestXray, SkinCancer, BrainTumor, CovidXray

**Audio**: SpeechCommands, Gtzan, Esc50, Urbansound8k

## Reward Distribution

- 90% → Trainers (proportional to improvement)
- 10% → Validators (split equally)
- 2% → Treasury fee

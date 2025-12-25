# DECLOUD Creator Kit

Create and manage federated learning rounds on DECLOUD.

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Setup

```bash
decloud-creator setup
```

This will configure:
- Network (devnet/mainnet)
- Wallet (private key)
- IPFS storage (Pinata or local node)

### 2. Create Training Round

Interactive:
```bash
decloud-creator create
```

Or with arguments:
```bash
decloud-creator create \
  --model model.pt \
  --dataset Cifar10 \
  --reward 0.01 \
  --private-key YOUR_KEY
```

### 3. Track Your Trainings

```bash
# List all
decloud-creator list

# Check status
decloud-creator status 42

# Watch in real-time
decloud-creator watch 42
```

### 4. Cancel Round

```bash
decloud-creator cancel 42
```

## Commands

| Command | Description |
|---------|-------------|
| `setup` | Initial configuration |
| `create` | Create new training round |
| `list` | List my trainings |
| `status <id>` | Check round status |
| `watch <id>` | Watch round in real-time |
| `cancel <id>` | Cancel and get refund |
| `upload <file>` | Upload model to IPFS |

## IPFS Setup

### Option 1: Pinata (Recommended)

1. Create free account at https://app.pinata.cloud
2. Generate API keys
3. Run `decloud-creator setup` and enter keys

### Option 2: Local IPFS Node

```bash
# Install
wget https://dist.ipfs.tech/kubo/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.24.0_linux-amd64.tar.gz
cd kubo && sudo ./install.sh

# Run daemon
ipfs init
ipfs daemon
```

## Python API

```python
from decloud_creator import Creator

creator = Creator()
creator.login("your_private_key")

# Upload model
cid = creator.upload_model("model.pt")

# Create round
round_id = creator.create_round(
    model="model.pt",      # or IPFS CID
    dataset="Cifar10",
    reward=0.01
)

# Check status
info = creator.get_round_status(round_id)
print(f"Status: {info.status}")

# List trainings
for r in creator.list_trainings():
    print(f"Round #{r.round_id}: {r.status}")

# Watch progress
creator.watch(round_id)
```

## Supported Datasets

**Images:**
- Cifar10, Cifar100, Mnist, FashionMnist
- Food101, Flowers102, StanfordDogs, etc.

**Text:**
- Imdb, Sst2, AgNews, Dbpedia, etc.

**And more!** Run `decloud-creator create` and type `?` to see all.

## Environment Variables

```bash
export DECLOUD_PRIVATE_KEY="your_key"
export PINATA_API_KEY="xxx"
export PINATA_SECRET_KEY="yyy"
```

## Round Lifecycle

```
Created → WaitingValidator → WaitingTrainers → Training → Validating → Completed
              ↓                    ↓                ↓            
           Expired             Expired           Expired
              ↓
          Cancelled (refund)
```

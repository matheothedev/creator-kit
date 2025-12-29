#!/usr/bin/env python3
"""
DECLOUD Creator CLI
===================

Command-line interface for creating and managing federated learning rounds.

Usage:
    decloud-creator setup           # Setup IPFS and wallet
    decloud-creator create          # Create new training round (interactive)
    decloud-creator list            # List my trainings
    decloud-creator status <id>     # Check round status
    decloud-creator watch <id>      # Watch round in real-time
    decloud-creator cancel <id>     # Cancel round and get refund
"""

import os
import sys
import argparse
from typing import Optional

from .config import Config, DATASETS, POPULAR_DATASETS
from .creator import Creator
from .storage import format_round_status
from .ipfs import setup_pinata_interactive


def get_private_key(args) -> str:
    """Get private key from args or env"""
    if hasattr(args, 'private_key') and args.private_key:
        return args.private_key
    
    key = os.getenv("DECLOUD_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY")
    if key:
        return key
    
    # Try to load from config
    config = Config.load()
    if config.private_key:
        return config.private_key
    
    print("❌ No private key provided!")
    print("   Use --private-key or set DECLOUD_PRIVATE_KEY env var")
    sys.exit(1)


def cmd_setup(args):
    """Setup command - configure IPFS and wallet"""
    print("═" * 60)
    print("  🛠️  DECLOUD Creator Setup")
    print("═" * 60)
    
    config = Config.load()
    
    # 1. Network selection
    print("\n1️⃣  Network")
    print("   Current:", config.network)
    choice = input("   Change? [devnet/mainnet-beta/skip]: ").strip().lower()
    if choice in ("devnet", "mainnet-beta"):
        config.network = choice
        if choice == "devnet":
            config.rpc_url = "https://api.devnet.solana.com"
        else:
            config.rpc_url = "https://api.mainnet-beta.solana.com"
        print(f"   ✓ Set to {choice}")
    
    # 2. Private key
    print("\n2️⃣  Wallet")
    if config.private_key:
        print("   ✓ Private key configured")
    else:
        key = input("   Enter private key (base58) or skip: ").strip()
        if key and len(key) > 40:
            config.private_key = key
            print("   ✓ Key saved")
    
    # 3. IPFS setup
    print("\n3️⃣  IPFS Storage")
    print("   Options:")
    print("   [1] Pinata (free 1GB) - recommended")
    print("   [2] Local IPFS node")
    print("   [3] Skip")
    
    choice = input("   Choose [1/2/3]: ").strip()
    
    if choice == "1":
        api_key, secret_key = setup_pinata_interactive()
        if api_key:
            config.pinata_api_key = api_key
            config.pinata_secret_key = secret_key
    elif choice == "2":
        print("   Make sure 'ipfs daemon' is running")
        print("   ✓ Will use local node at localhost:5001")
    
    # Save config
    config.save()
    
    print("\n" + "═" * 60)
    print("  ✅ Setup complete!")
    print("═" * 60)
    print("\nNext steps:")
    print("  decloud-creator create    # Create a training round")
    print("  decloud-creator list      # View your trainings")


def cmd_create(args):
    """Create command - create new training round"""
    config = Config.load()
    creator = Creator(config)
    
    # Login
    private_key = get_private_key(args)
    pubkey, balance = creator.login(private_key)
    
    print("═" * 60)
    print("  🚀 Create Training Round")
    print("═" * 60)
    print(f"  Wallet:  {pubkey[:20]}...")
    print(f"  Balance: {balance:.4f} SOL")
    print("═" * 60)
    
    # Interactive mode if no args
    if not args.model:
        print("\n1️⃣  Model")
        print("   Enter path to model file OR IPFS CID")
        model = input("   > ").strip()
    else:
        model = args.model
    
    if not args.dataset:
        print("\n2️⃣  Dataset")
        print("   Popular options:")
        for i, ds in enumerate(POPULAR_DATASETS, 1):
            print(f"   [{i}] {ds}")
        print(f"   [?] Show all ({len(DATASETS)} available)")
        
        choice = input("   > ").strip()
        
        if choice == "?":
            print("\n   All datasets:")
            for i, ds in enumerate(DATASETS, 1):
                print(f"   {ds}", end="  ")
                if i % 5 == 0:
                    print()
            print()
            dataset = input("   Enter dataset name: ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(POPULAR_DATASETS):
            dataset = POPULAR_DATASETS[int(choice) - 1]
        else:
            dataset = choice
    else:
        dataset = args.dataset
    
    if not args.reward:
        print("\n3️⃣  Reward")
        print(f"   How much SOL to pay for training?")
        print(f"   (Your balance: {balance:.4f} SOL)")
        reward_str = input("   > ").strip()
        reward = float(reward_str)
    else:
        reward = args.reward
    
    # Confirm
    print("\n" + "─" * 60)
    print("  Summary:")
    print(f"    Model:   {model[:50]}{'...' if len(model) > 50 else ''}")
    print(f"    Dataset: {dataset}")
    print(f"    Reward:  {reward} SOL")
    print("─" * 60)
    
    if not args.yes:
        try:
            confirm = input("\n  Create round? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("  Cancelled.")
                return
        except (UnicodeDecodeError, EOFError):
            print("  Cancelled (input error).")
            return
    
    print()
    
    try:
        round_id = creator.create_round(
            model=model,
            dataset=dataset,
            reward=reward,
        )
        
        print("\n" + "═" * 60)
        print(f"  ✅ Round #{round_id} created!")
        print("═" * 60)
        print(f"  Status: Waiting for validator")
        print(f"  View:   decloud-creator status {round_id}")
        print(f"  Watch:  decloud-creator watch {round_id}")
        print(f"  Cancel: decloud-creator cancel {round_id}")
        
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        sys.exit(1)


def cmd_list(args):
    """List command - show my trainings"""
    config = Config.load()
    creator = Creator(config)
    
    # Login if key provided (for refresh)
    if hasattr(args, 'private_key') and args.private_key:
        creator.login(args.private_key)
    
    print("═" * 60)
    print("  📋 My Trainings")
    print("═" * 60)
    
    rounds = creator.list_trainings(refresh=args.refresh)
    
    if not rounds:
        print("\n  No trainings yet.")
        print("  Create one: decloud-creator create")
        return
    
    print()
    for r in rounds:
        print(f"  {format_round_status(r)}")
    
    print()
    
    # Stats
    stats = creator.get_stats()
    print("─" * 60)
    print(f"  Total: {stats['total_rounds']} rounds | Spent: {stats['total_spent_sol']:.4f} SOL")
    
    by_status = stats.get('by_status', {})
    if by_status:
        status_str = " | ".join(f"{k}: {v}" for k, v in by_status.items())
        print(f"  {status_str}")


def cmd_status(args):
    """Status command - check round status"""
    config = Config.load()
    creator = Creator(config)
    
    try:
        info = creator.get_round_status(args.round_id)
        
        print("═" * 60)
        print(f"  Round #{info.id}")
        print("═" * 60)
        
        status_icons = {
            "waitingValidator": "⏳",
            "waitingTrainers": "👥",
            "training": "🏋️",
            "validating": "🔬",
            "completed": "✅",
            "expired": "⏰",
            "cancelled": "❌",
        }
        icon = status_icons.get(info.status, "❓")
        
        print(f"  Status:     {icon} {info.status}")
        print(f"  Dataset:    {info.dataset}")
        print(f"  Reward:     {info.reward_amount / 1e9:.4f} SOL")
        print(f"  Model CID:  {info.model_cid}")
        
        if info.validator and info.validator != "11111111111111111111111111111111":
            print(f"  Validator:  {info.validator[:20]}...")
        
        if info.trainers_count > 0:
            print(f"  Progress:   {info.submissions_count}/{info.trainers_count} submitted")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_watch(args):
    """Watch command - watch round in real-time"""
    config = Config.load()
    creator = Creator(config)
    
    try:
        creator.watch(args.round_id, interval=args.interval)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def cmd_cancel(args):
    """Cancel command - cancel round and get refund"""
    config = Config.load()
    creator = Creator(config)
    
    private_key = get_private_key(args)
    pubkey, balance = creator.login(private_key)
    
    print(f"  Wallet: {pubkey[:20]}...")
    print(f"  Cancelling round #{args.round_id}...")
    
    try:
        # Check status first
        info = creator.get_round_status(args.round_id)
        
        if info.status != "waitingValidator":
            print(f"  ❌ Cannot cancel: round is in '{info.status}' status")
            print("     Only rounds waiting for validator can be cancelled.")
            sys.exit(1)
        
        if not args.yes:
            confirm = input(f"  Cancel and refund {info.reward_amount/1e9:.4f} SOL? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("  Aborted.")
                return
        
        tx = creator.cancel_round(args.round_id)
        print(f"  ✅ Cancelled! TX: {tx[:40]}...")
        print(f"  Refunded: {info.reward_amount/1e9:.4f} SOL")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)


def cmd_upload(args):
    """Upload model to IPFS"""
    config = Config.load()
    creator = Creator(config)
    
    print(f"  📤 Uploading {args.file}...")
    
    try:
        cid = creator.upload_model(args.file)
        print(f"  ✅ Uploaded!")
        print(f"  CID: {cid}")
        print(f"  URL: https://ipfs.io/ipfs/{cid}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="DECLOUD Creator - Create federated learning rounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  decloud-creator setup                    # Initial setup
  decloud-creator create                   # Create round (interactive)
  decloud-creator create -m model.pt -d Cifar10 -r 0.01
  decloud-creator list                     # List my trainings
  decloud-creator status 42                # Check round #42
  decloud-creator watch 42                 # Watch round #42
  decloud-creator cancel 42                # Cancel round #42
  decloud-creator upload model.pt          # Upload to IPFS
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Setup
    setup_parser = subparsers.add_parser("setup", help="Setup IPFS and wallet")
    
    # Create
    create_parser = subparsers.add_parser("create", help="Create training round")
    create_parser.add_argument("-m", "--model", help="Model file path or IPFS CID")
    create_parser.add_argument("-d", "--dataset", help="Dataset name")
    create_parser.add_argument("-r", "--reward", type=float, help="Reward in SOL")
    create_parser.add_argument("-k", "--private-key", help="Wallet private key")
    create_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # List
    list_parser = subparsers.add_parser("list", help="List my trainings")
    list_parser.add_argument("--refresh", action="store_true", help="Refresh status from blockchain")
    list_parser.add_argument("-k", "--private-key", help="Wallet private key (for refresh)")
    
    # Status
    status_parser = subparsers.add_parser("status", help="Check round status")
    status_parser.add_argument("round_id", type=int, help="Round ID")
    
    # Watch
    watch_parser = subparsers.add_parser("watch", help="Watch round in real-time")
    watch_parser.add_argument("round_id", type=int, help="Round ID")
    watch_parser.add_argument("-i", "--interval", type=int, default=5, help="Poll interval (seconds)")
    
    # Cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel round and get refund")
    cancel_parser.add_argument("round_id", type=int, help="Round ID")
    cancel_parser.add_argument("-k", "--private-key", help="Wallet private key")
    cancel_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # Upload
    upload_parser = subparsers.add_parser("upload", help="Upload model to IPFS")
    upload_parser.add_argument("file", help="File to upload")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "setup": cmd_setup,
        "create": cmd_create,
        "list": cmd_list,
        "status": cmd_status,
        "watch": cmd_watch,
        "cancel": cmd_cancel,
        "upload": cmd_upload,
    }
    
    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
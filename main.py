#!/usr/bin/env python3
"""
Decloud Creator CLI
Create training rounds and manage model packages
"""
import sys
import asyncio
import getpass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from config import config, DATASETS, PACKAGES_DIR
from creator import DeCloudCreator
from lighthouse_client import LighthouseClient, get_lighthouse_client, init_lighthouse_client

console = Console()


def get_creator() -> DeCloudCreator:
    """Get creator instance"""
    if not config.private_key:
        console.print("[red]No private key configured. Run 'decloud-creator setup' first.[/red]")
        sys.exit(1)
    return DeCloudCreator(config.private_key)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Decloud Creator CLI
    
    Create training rounds and earn from federated learning on Solana.
    """
    pass


# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

@cli.command()
def setup():
    """Interactive setup wizard"""
    console.print("\n[bold cyan]=== Decloud Creator Setup ===[/bold cyan]\n")
    
    # Private key
    console.print("[yellow]Enter your Solana wallet private key (base58)[/yellow]")
    private_key = getpass.getpass("Private Key: ")
    
    if not private_key:
        console.print("[red]Private key required[/red]")
        return
    
    try:
        from solana_client import SolanaClient
        client = SolanaClient.from_private_key(private_key)
        console.print(f"[green]✓ Wallet: {client.pubkey}[/green]")
    except Exception as e:
        console.print(f"[red]Invalid private key: {e}[/red]")
        return
    
    # Network
    console.print("\n[yellow]Select network:[/yellow]")
    for i, net in enumerate(["devnet", "mainnet", "testnet"], 1):
        console.print(f"  {i}. {net}")
    
    network_choice = Prompt.ask("Network", choices=["1", "2", "3"], default="1")
    network = ["devnet", "mainnet", "testnet"][int(network_choice) - 1]
    
    # Create Lighthouse API key automatically from Solana private key
    console.print("\n[yellow]Creating Lighthouse Storage API key...[/yellow]")
    console.print("[dim]Using your Solana wallet for IPFS uploads[/dim]")
    
    lighthouse_api_key = None
    try:
        lighthouse_api_key = LighthouseClient.create_api_key_from_private_key(
            private_key, 
            key_name=f"decloud-creator-{str(client.pubkey)[:8]}"
        )
        console.print(f"[green]✓ Lighthouse API key created![/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to create Lighthouse API key: {e}[/red]")
        console.print("[dim]You can create it manually later[/dim]")
    
    # Save
    config.private_key = private_key
    config.network = network
    config.lighthouse_api_key = lighthouse_api_key
    config.save()
    
    console.print(f"\n[green]✓ Configuration saved![/green]")
    
    # Test Lighthouse
    if config.has_lighthouse():
        console.print("[dim]Testing Lighthouse connection...[/dim]")
        lighthouse = init_lighthouse_client(config.lighthouse_api_key)
        if lighthouse.test_authentication_sync():
            console.print("[green]✓ Lighthouse Storage connected![/green]")
            
            # Show usage info
            usage = lighthouse.get_data_usage()
            if usage:
                console.print(f"[dim]  Data usage: {usage}[/dim]")
        else:
            console.print("[red]✗ Lighthouse authentication failed[/red]")
    
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Build package: [bold]decloud-creator build -m model.pt -d Cifar10[/bold]")
    console.print("  2. Create round: [bold]decloud-creator create -c <CID> -d Cifar10 -r 0.1[/bold]")


@cli.command()
@click.option("--network", "-n", type=click.Choice(["devnet", "mainnet", "testnet"]))
def network(network):
    """Change or show network"""
    if network:
        config.network = network
        config.save()
        console.print(f"[green]✓ Network: {network}[/green]")
    else:
        console.print(f"Network: [cyan]{config.network}[/cyan]")
        console.print(f"RPC: [dim]{config.rpc_url}[/dim]")


@cli.group()
def rpc():
    """RPC endpoint configuration"""
    pass


@rpc.command("set")
@click.argument("url")
def rpc_set(url):
    """Set custom RPC endpoint"""
    config.custom_rpc = url
    config.save()
    console.print(f"[green]✓ Custom RPC set: {url}[/green]")


@rpc.command("reset")
def rpc_reset():
    """Reset to default RPC (based on network)"""
    config.custom_rpc = None
    config.save()
    console.print(f"[green]✓ Reset to default RPC[/green]")
    console.print(f"[dim]Using: {config.rpc_url}[/dim]")


@rpc.command("show")
def rpc_show():
    """Show current RPC endpoint"""
    if config.custom_rpc:
        console.print(f"Custom RPC: [cyan]{config.custom_rpc}[/cyan]")
    else:
        console.print(f"Default RPC ({config.network}): [cyan]{config.rpc_url}[/cyan]")


# ═══════════════════════════════════════════════════════════════
# Package Building
# ═══════════════════════════════════════════════════════════════

@cli.command()
@click.option("--model", "-m", required=True, help="Path to PyTorch model (.pt/.pth)")
@click.option("--dataset", "-d", required=True, help="Dataset name (e.g., Cifar10)")
@click.option("--output", "-o", help="Output directory")
@click.option("--upload", "-u", is_flag=True, help="Upload to IPFS after building")
def build(model, dataset, output, upload):
    """Build model package from local model"""
    
    if dataset not in DATASETS:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print(f"[dim]Available: {', '.join(list(DATASETS.keys())[:10])}...[/dim]")
        return
    
    if not Path(model).exists():
        console.print(f"[red]Model file not found: {model}[/red]")
        return
    
    creator = get_creator()
    package_path = creator.build_package(model, dataset)
    
    if not package_path:
        console.print("[red]✗ Package build failed[/red]")
        return
    
    console.print(f"\n[green]✓ Package built: {package_path}[/green]")
    
    if upload:
        if not config.has_lighthouse():
            console.print("[red]Lighthouse not configured. Run setup first.[/red]")
            return
        
        cid = creator.upload_package_sync(package_path, dataset)
        if cid:
            console.print(f"\n[bold cyan]CID: {cid}[/bold cyan]")
            console.print(f"\n[dim]Create round: decloud-creator create -c {cid} -d {dataset} -r <reward_sol>[/dim]")


@cli.command()
@click.option("--path", "-p", required=True, help="Path to package directory")
@click.option("--dataset", "-d", required=True, help="Dataset name")
def upload(path, dataset):
    """Upload existing package to Lighthouse (IPFS)"""
    
    if not config.has_lighthouse():
        console.print("[red]Lighthouse not configured. Run setup first.[/red]")
        return
    
    package_path = Path(path)
    if not package_path.exists():
        console.print(f"[red]Directory not found: {path}[/red]")
        return
    
    if not (package_path / "config.json").exists():
        console.print(f"[red]Not a valid package (no config.json)[/red]")
        return
    
    creator = get_creator()
    cid = creator.upload_package_sync(package_path, dataset)
    
    if cid:
        console.print(f"\n[bold cyan]CID: {cid}[/bold cyan]")
        console.print(f"\n[dim]Create round: decloud-creator create -c {cid} -d {dataset} -r <reward_sol>[/dim]")


# ═══════════════════════════════════════════════════════════════
# Round Management
# ═══════════════════════════════════════════════════════════════

@cli.command()
@click.option("--cid", "-c", required=True, help="IPFS CID of model package")
@click.option("--dataset", "-d", required=True, help="Dataset name")
@click.option("--reward", "-r", required=True, type=float, help="Reward in SOL")
@click.option("--min-rating", "-m", default=5.0, type=float, help="Min trainer rating (1.0-5.0 stars, default: 5.0 = no restriction)")
def create(cid, dataset, reward, min_rating):
    """Create a new training round"""
    
    if dataset not in DATASETS:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        return
    
    if reward <= 0:
        console.print(f"[red]Reward must be positive[/red]")
        return
    
    if min_rating < 1.0 or min_rating > 5.0:
        console.print(f"[red]Min rating must be between 1.0 and 5.0[/red]")
        return
    
    creator = get_creator()
    
    # Check balance
    balance = creator.get_balance()
    if balance < reward + 0.01:
        console.print(f"[red]Insufficient balance: {balance:.4f} SOL[/red]")
        return
    
    result = creator.create_round(cid, dataset, reward, min_rating)
    
    if result.get("success"):
        console.print(f"\n[bold green]✓ Round #{result['round_id']} created![/bold green]")
    else:
        console.print(f"[red]✗ {result.get('error')}[/red]")


@cli.command()
@click.argument("round_id", type=int)
def finalize(round_id):
    """Finalize a training round"""
    creator = get_creator()
    result = creator.finalize_round(round_id)
    
    if not result.get("success"):
        console.print(f"[red]✗ {result.get('error')}[/red]")


@cli.command("force-finalize")
@click.argument("round_id", type=int)
def force_finalize_cmd(round_id):
    """Force finalize after 12h deadline"""
    creator = get_creator()
    result = creator.force_finalize(round_id)
    
    if not result.get("success"):
        console.print(f"[red]✗ {result.get('error')}[/red]")


@cli.command()
@click.argument("round_id", type=int)
def cancel(round_id):
    """Cancel a round (only if no participants)"""
    creator = get_creator()
    result = creator.cancel_round(round_id)
    
    if not result.get("success"):
        console.print(f"[red]✗ {result.get('error')}[/red]")


@cli.command()
@click.argument("round_id", type=int)
def withdraw(round_id):
    """Withdraw remaining funds from finalized round"""
    creator = get_creator()
    result = creator.withdraw_remainder(round_id)
    
    if not result.get("success"):
        console.print(f"[red]✗ {result.get('error')}[/red]")


# ═══════════════════════════════════════════════════════════════
# Downloads
# ═══════════════════════════════════════════════════════════════

@cli.command("download-gradient")
@click.argument("cid")
def download_gradient(cid):
    """Download gradient package from IPFS"""
    creator = get_creator()
    path = creator.download_gradient_sync(cid)
    
    if path:
        console.print(f"\n[dim]Files in {path}:[/dim]")
        for f in path.iterdir():
            console.print(f"  {f.name}")


@cli.command("download-model")
@click.argument("cid")
def download_model(cid):
    """Download base model package from IPFS"""
    creator = get_creator()
    path = creator.download_model_sync(cid)
    
    if path:
        console.print(f"\n[dim]Files in {path}:[/dim]")
        for f in path.iterdir():
            console.print(f"  {f.name}")


# ═══════════════════════════════════════════════════════════════
# Status & Info
# ═══════════════════════════════════════════════════════════════

@cli.command()
def status():
    """Show creator status"""
    creator = get_creator()
    creator.show_status()


@cli.command("balance")
def show_balance():
    """Show wallet balance"""
    creator = get_creator()
    try:
        balance = creator.get_balance()
        console.print(f"Balance: [green]{balance:.6f} SOL[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("my-rounds")
@click.option("--limit", "-l", default=10)
def my_rounds(limit):
    """Show rounds created by you"""
    creator = get_creator()
    creator.show_my_rounds(limit=limit)


@cli.command("info")
@click.argument("round_id", type=int)
def round_info(round_id):
    """Show detailed round info"""
    creator = get_creator()
    creator.show_round_details(round_id)


@cli.command("datasets")
def list_datasets():
    """List available datasets"""
    console.print("[bold]Available Datasets:[/bold]\n")
    
    categories = {
        "Image": ["Cifar10", "Cifar100", "Mnist", "FashionMnist", "Emnist", "Kmnist", 
                  "Food101", "Flowers102", "Svhn", "Caltech101", "Eurosat"],
        "Text": ["Imdb", "Sst2", "AgNews", "Dbpedia", "YelpReviews", "AmazonPolarity"],
        "Tabular": ["Iris", "Wine", "Diabetes", "BreastCancer", "CaliforniaHousing"],
        "Medical": ["ChestXray", "SkinCancer", "BrainTumor", "CovidXray"],
        "Audio": ["SpeechCommands", "Gtzan", "Esc50", "Urbansound8k"],
    }
    
    for cat, datasets in categories.items():
        console.print(f"[cyan]{cat}:[/cyan]")
        for ds in datasets:
            console.print(f"  {ds}")
        console.print()


# ═══════════════════════════════════════════════════════════════
# Full Workflow
# ═══════════════════════════════════════════════════════════════

@cli.command()
@click.option("--model", "-m", required=True, help="Path to model")
@click.option("--dataset", "-d", required=True, help="Dataset name")
@click.option("--reward", "-r", required=True, type=float, help="Reward in SOL")
@click.option("--min-rating", type=float, default=5.0, help="Min trainer rating (1.0-5.0 stars)")
def launch(model, dataset, reward, min_rating):
    """Build, upload and create round in one command"""
    
    if dataset not in DATASETS:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        return
    
    if not Path(model).exists():
        console.print(f"[red]Model not found: {model}[/red]")
        return
    
    if not config.has_lighthouse():
        console.print("[red]Lighthouse not configured. Run setup first.[/red]")
        return
    
    if min_rating < 1.0 or min_rating > 5.0:
        console.print(f"[red]Min rating must be between 1.0 and 5.0[/red]")
        return
    
    creator = get_creator()
    
    # Check balance
    balance = creator.get_balance()
    if balance < reward + 0.01:
        console.print(f"[red]Insufficient balance: {balance:.4f} SOL[/red]")
        return
    
    # 1. Build
    console.print("\n[bold cyan]Step 1: Building package...[/bold cyan]")
    package_path = creator.build_package(model, dataset)
    
    if not package_path:
        console.print("[red]✗ Build failed[/red]")
        return
    
    # 2. Upload
    console.print("\n[bold cyan]Step 2: Uploading to IPFS...[/bold cyan]")
    cid = creator.upload_package_sync(package_path, dataset)
    
    if not cid:
        console.print("[red]✗ Upload failed[/red]")
        return
    
    # 3. Create round
    console.print("\n[bold cyan]Step 3: Creating round...[/bold cyan]")
    result = creator.create_round(cid, dataset, reward, min_rating)
    
    if result.get("success"):
        console.print(f"\n{'═'*50}")
        console.print(f"[bold green]✓ Round #{result['round_id']} launched![/bold green]")
        console.print(f"[dim]  Dataset: {dataset}[/dim]")
        console.print(f"[dim]  Reward: {reward} SOL[/dim]")
        console.print(f"[dim]  Min Rating: {min_rating:.2f} ★[/dim]")
        console.print(f"[dim]  CID: {cid}[/dim]")
        console.print(f"\n[cyan]Waiting for validators and trainers...[/cyan]")
        console.print(f"[dim]Check status: decloud-creator info {result['round_id']}[/dim]")
    else:
        console.print(f"[red]✗ {result.get('error')}[/red]")


def main():
    cli()


if __name__ == "__main__":
    main()
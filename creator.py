"""
Core Decloud Creator - Round management
"""
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from config import config, PACKAGES_DIR
from solana_client import SolanaClient, RoundInfo, GradientInfo
from ipfs_client import ipfs_client
from pinata_client import pinata_client
from model_builder import create_package

console = Console()


@dataclass
class CreatorStats:
    """Creator statistics"""
    rounds_created: int = 0
    rounds_finalized: int = 0
    total_rewards: float = 0


class DeCloudCreator:
    """
    Decloud Creator - Create and manage training rounds
    """
    
    def __init__(self, private_key: str):
        self.solana = SolanaClient.from_private_key(private_key)
        self.stats = CreatorStats()
        
        console.print(f"[green]✓ Creator initialized[/green]")
        console.print(f"[dim]  Wallet: {self.solana.pubkey}[/dim]")
        console.print(f"[dim]  Network: {config.network}[/dim]")
    
    def get_balance(self) -> float:
        """Get SOL balance"""
        return self.solana.get_balance() / 1e9
    
    # ═══════════════════════════════════════════════════════════════
    # Package Creation
    # ═══════════════════════════════════════════════════════════════
    
    def build_package(self, model_path: str, dataset: str) -> Optional[Path]:
        """Build model package from local model"""
        return create_package(model_path, dataset)
    
    async def upload_package(self, package_dir: Path, dataset: str) -> Optional[str]:
        """Upload package to IPFS via Pinata"""
        console.print(f"[cyan]Uploading to IPFS...[/cyan]")
        cid = await pinata_client.upload_model_package(package_dir, dataset)
        
        if cid:
            console.print(f"[green]✓ Uploaded: {cid}[/green]")
        else:
            console.print(f"[red]✗ Upload failed[/red]")
        
        return cid
    
    def upload_package_sync(self, package_dir: Path, dataset: str) -> Optional[str]:
        """Sync wrapper"""
        return asyncio.run(self.upload_package(package_dir, dataset))
    
    # ═══════════════════════════════════════════════════════════════
    # Round Management
    # ═══════════════════════════════════════════════════════════════
    
    def create_round(
        self,
        model_cid: str,
        dataset: str,
        reward_sol: float,
        min_trainer_rating: float = 5.0,  # NEW: minimum rating (1.0 - 5.0 stars)
    ) -> Dict[str, Any]:
        """
        Create a new training round
        
        Args:
            model_cid: IPFS CID of model package
            dataset: Dataset name
            reward_sol: Reward amount in SOL
            min_trainer_rating: Minimum trainer rating (1.0 - 5.0 stars, default 5.0 = no restriction)
        
        Returns:
            {"success": bool, "round_id": int, "tx": str} or {"error": str}
        """
        try:
            reward_lamports = int(reward_sol * 1e9)
            # Convert float rating (1.0-5.0) to internal format (100-500)
            min_rating_internal = int(min_trainer_rating * 100)
            
            if min_rating_internal < 100 or min_rating_internal > 500:
                return {"error": "min_trainer_rating must be between 1.0 and 5.0"}
            
            console.print(f"[cyan]Creating round...[/cyan]")
            console.print(f"[dim]  Dataset: {dataset}[/dim]")
            console.print(f"[dim]  Reward: {reward_sol} SOL[/dim]")
            console.print(f"[dim]  Min Trainer Rating: {min_trainer_rating:.2f} ★[/dim]")
            console.print(f"[dim]  Model CID: {model_cid[:30]}...[/dim]")
            
            tx, round_id = self.solana.create_round(
                model_cid, dataset, reward_lamports, min_rating_internal
            )
            
            # Track locally
            config.track_round(round_id, model_cid)
            self.stats.rounds_created += 1
            
            console.print(f"[green]✓ Round #{round_id} created![/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return {"success": True, "round_id": round_id, "tx": tx}
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return {"error": str(e)}
    
    def finalize_round(self, round_id: int) -> Dict[str, Any]:
        """Finalize a round"""
        try:
            round_info = self.solana.get_round(round_id)
            
            if not round_info:
                return {"error": "Round not found"}
            
            if round_info.status != "Active":
                return {"error": f"Round not active: {round_info.status}"}
            
            if round_info.pre_count == 0:
                return {"error": "No prevalidation yet"}
            
            if round_info.gradients_count == 0:
                return {"error": "No gradients submitted"}
            
            if round_info.total_validations <= round_info.pre_count:
                return {"error": "No postvalidation yet"}
            
            console.print(f"[cyan]Finalizing round #{round_id}...[/cyan]")
            tx = self.solana.finalize_round(round_id)
            
            self.stats.rounds_finalized += 1
            
            console.print(f"[green]✓ Round #{round_id} finalized![/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return {"success": True, "tx": tx}
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return {"error": str(e)}
    
    def force_finalize(self, round_id: int) -> Dict[str, Any]:
        """Force finalize after deadline"""
        try:
            console.print(f"[cyan]Force finalizing round #{round_id}...[/cyan]")
            tx = self.solana.force_finalize(round_id)
            
            console.print(f"[green]✓ Round #{round_id} finalized![/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return {"success": True, "tx": tx}
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return {"error": str(e)}
    
    def cancel_round(self, round_id: int) -> Dict[str, Any]:
        """Cancel a round (only if no participants)"""
        try:
            round_info = self.solana.get_round(round_id)
            
            if not round_info:
                return {"error": "Round not found"}
            
            if round_info.status != "Active":
                return {"error": f"Round not active: {round_info.status}"}
            
            if round_info.pre_count > 0 or round_info.gradients_count > 0:
                return {"error": "Cannot cancel - round has participants"}
            
            console.print(f"[cyan]Cancelling round #{round_id}...[/cyan]")
            tx = self.solana.cancel_round(round_id)
            
            console.print(f"[green]✓ Round #{round_id} cancelled! Funds refunded.[/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return {"success": True, "tx": tx}
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return {"error": str(e)}
    
    def withdraw_remainder(self, round_id: int) -> Dict[str, Any]:
        """Withdraw remaining funds from finalized round"""
        try:
            round_info = self.solana.get_round(round_id)
            
            if not round_info:
                return {"error": "Round not found"}
            
            if round_info.status != "Finalized":
                return {"error": f"Round not finalized: {round_info.status}"}
            
            console.print(f"[cyan]Withdrawing remainder from round #{round_id}...[/cyan]")
            tx = self.solana.withdraw_remainder(round_id)
            
            console.print(f"[green]✓ Remainder withdrawn![/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return {"success": True, "tx": tx}
            
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════
    # Download Gradients
    # ═══════════════════════════════════════════════════════════════
    
    async def download_gradient(self, gradient_cid: str) -> Optional[Path]:
        """Download gradient package from IPFS"""
        console.print(f"[cyan]Downloading gradient {gradient_cid[:20]}...[/cyan]")
        path = await ipfs_client.download_gradient(gradient_cid)
        
        if path:
            console.print(f"[green]✓ Downloaded to {path}[/green]")
        else:
            console.print(f"[red]✗ Download failed[/red]")
        
        return path
    
    def download_gradient_sync(self, gradient_cid: str) -> Optional[Path]:
        """Sync wrapper"""
        return asyncio.run(self.download_gradient(gradient_cid))
    
    async def download_base_model(self, model_cid: str) -> Optional[Path]:
        """Download base model package from IPFS"""
        console.print(f"[cyan]Downloading model {model_cid[:20]}...[/cyan]")
        path = await ipfs_client.download_package(model_cid)
        
        if path:
            console.print(f"[green]✓ Downloaded to {path}[/green]")
        else:
            console.print(f"[red]✗ Download failed[/red]")
        
        return path
    
    def download_model_sync(self, model_cid: str) -> Optional[Path]:
        """Sync wrapper"""
        return asyncio.run(self.download_base_model(model_cid))
    
    # ═══════════════════════════════════════════════════════════════
    # Status & Info
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_info(self, round_id: int) -> Optional[RoundInfo]:
        """Get round details"""
        return self.solana.get_round(round_id)
    
    def get_my_rounds(self) -> List[RoundInfo]:
        """Get all rounds created by this wallet"""
        return self.solana.get_my_rounds()
    
    def show_status(self):
        """Display creator status"""
        try:
            balance = self.get_balance()
        except:
            balance = -1
        
        table = Table(title="🎨 Creator Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Wallet", str(self.solana.pubkey))
        table.add_row("Balance", f"{balance:.4f} SOL" if balance >= 0 else "Error")
        table.add_row("Network", config.network)
        table.add_row("Rounds Created", str(self.stats.rounds_created))
        table.add_row("Rounds Finalized", str(self.stats.rounds_finalized))
        table.add_row("Pinata", "✓ Configured" if config.has_pinata() else "✗ Not configured")
        
        console.print(table)
    
    def show_my_rounds(self, limit: int = 10):
        """Display rounds created by this wallet"""
        rounds = self.get_my_rounds()
        
        if not rounds:
            console.print("[yellow]No rounds created yet[/yellow]")
            return
        
        table = Table(title=f"My Rounds ({len(rounds)} total)")
        table.add_column("ID", style="cyan")
        table.add_column("Dataset", style="yellow")
        table.add_column("Reward", style="green")
        table.add_column("Min ★", style="magenta")
        table.add_column("Pre", style="blue")
        table.add_column("Trainers", style="white")
        table.add_column("Status", style="white")
        
        for round_info in rounds[:limit]:
            reward_sol = round_info.reward_amount / 1e9
            min_rating = round_info.min_trainer_rating / 100
            
            status_colors = {
                "Active": "[yellow]Active[/yellow]",
                "Finalized": "[green]Finalized[/green]",
                "Cancelled": "[red]Cancelled[/red]",
            }
            status = status_colors.get(round_info.status, round_info.status)
            
            table.add_row(
                str(round_info.id),
                round_info.dataset,
                f"{reward_sol:.4f}",
                f"{min_rating:.2f}",
                str(round_info.pre_count),
                str(round_info.gradients_count),
                status,
            )
        
        console.print(table)
    
    def show_round_details(self, round_id: int):
        """Display detailed round info"""
        info = self.get_round_info(round_id)
        
        if not info:
            console.print(f"[red]Round {round_id} not found[/red]")
            return
        
        table = Table(title=f"Round #{round_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Dataset", info.dataset)
        table.add_row("Reward", f"{info.reward_amount / 1e9:.4f} SOL")
        table.add_row("Min Trainer Rating", f"{info.min_trainer_rating / 100:.2f} ★")
        table.add_row("Status", info.status)
        table.add_row("Pre-validators", str(info.pre_count))
        table.add_row("Avg Pre Accuracy", f"{info.pre_accuracy_sum / max(1, info.pre_count) / 100:.2f}%" if info.pre_count > 0 else "-")
        table.add_row("Trainers", str(info.gradients_count))
        table.add_row("Total Validations", str(info.total_validations))
        table.add_row("Total Improvement", str(info.total_improvement))
        if info.status == "Finalized":
            table.add_row("Consensus Accuracy", f"{info.consensus_accuracy / 100:.2f}%")
        table.add_row("Model CID", info.model_cid)
        table.add_row("Creator", info.creator[:20] + "...")
        
        console.print(table)
        
        # Can finalize?
        if info.status == "Active":
            can_finalize = (
                info.pre_count > 0 and
                info.gradients_count > 0 and
                info.total_validations > info.pre_count
            )
            
            if can_finalize:
                console.print(f"\n[green]✓ Ready to finalize![/green]")
                console.print(f"[dim]Run: decloud-creator finalize {round_id}[/dim]")
            else:
                reasons = []
                if info.pre_count == 0:
                    reasons.append("waiting for prevalidation")
                if info.gradients_count == 0:
                    reasons.append("waiting for trainers")
                if info.total_validations <= info.pre_count:
                    reasons.append("waiting for postvalidation")
                console.print(f"\n[yellow]⏳ Cannot finalize yet: {', '.join(reasons)}[/yellow]")
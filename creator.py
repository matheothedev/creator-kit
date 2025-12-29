"""
DECLOUD Creator - Main Module
==============================

Create and manage federated learning rounds.
"""

import os
import subprocess
import json
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .config import Config, DATASETS, POPULAR_DATASETS
from .ipfs import IPFSUploader
from .storage import RoundStorage, StoredRound, format_round_status


@dataclass
class RoundInfo:
    """Round information from blockchain"""
    id: int
    creator: str
    model_cid: str
    dataset: str
    reward_amount: int
    status: str
    validator: str
    trainers_count: int
    submissions_count: int


class Creator:
    """
    DECLOUD Creator - create and manage training rounds.
    
    Usage:
        creator = Creator()
        creator.login("your_private_key")
        
        # Create a round
        round_id = creator.create_round(
            model="model.pt",  # or IPFS CID
            dataset="Cifar10",
            reward=0.01
        )
        
        # Check status
        creator.get_status(round_id)
        
        # List my trainings
        creator.list_trainings()
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.storage = RoundStorage(self.config.data_dir)
        self.ipfs = IPFSUploader(
            pinata_api_key=self.config.pinata_api_key,
            pinata_secret_key=self.config.pinata_secret_key,
        )
        
        self._private_key: Optional[str] = None
        self._public_key: Optional[str] = None
    
    def login(self, private_key: str) -> Tuple[str, float]:
        """
        Login with private key.
        
        Returns:
            Tuple of (public_key, balance_in_sol)
        """
        self._private_key = private_key
        
        # Get public key and balance
        script = f'''
const {{ Keypair, Connection, LAMPORTS_PER_SOL }} = require("@solana/web3.js");
const bs58 = require("bs58");

const keypair = Keypair.fromSecretKey(bs58.decode("{private_key}"));
const conn = new Connection("{self.config.rpc_url}", "confirmed");

(async () => {{
    const balance = await conn.getBalance(keypair.publicKey);
    console.log(JSON.stringify({{
        publicKey: keypair.publicKey.toBase58(),
        balance: balance
    }}));
}})();
'''
        result = self._run_node(script)
        data = json.loads(result)
        
        self._public_key = data["publicKey"]
        balance_sol = data["balance"] / 1e9
        
        return self._public_key, balance_sol
    
    def _run_node(self, script: str) -> str:
        """Run Node.js script"""
        result = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        
        return result.stdout.strip()
    
    def get_next_round_id(self) -> int:
        """Get next available round ID"""
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey }} = require("@solana/web3.js");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));

(async () => {{
    const provider = new anchor.AnchorProvider(conn, {{ publicKey: PublicKey.default }}, {{}});
    const program = new anchor.Program(idl, provider);
    
    const [counterPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round_counter")],
        programId
    );
    
    try {{
        const counter = await program.account.roundCounter.fetch(counterPda);
        console.log(counter.count.toString());
    }} catch (e) {{
        console.log("0");
    }}
}})();
'''
        result = self._run_node(script)
        return int(result)
    
    def upload_model(self, model_path: str) -> str:
        """
        Upload model to IPFS.
        
        Args:
            model_path: Path to model file
        
        Returns:
            IPFS CID
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        result = self.ipfs.upload(model_path)
        return result.cid
    
    def create_round(
        self,
        model: str,
        dataset: str,
        reward: float,
    ) -> int:
        """
        Create a new training round.
        
        Args:
            model: Path to model file OR IPFS CID
            dataset: Dataset name (e.g. "Cifar10")
            reward: Reward amount in SOL
        
        Returns:
            Round ID
        """
        if not self._private_key:
            raise RuntimeError("Not logged in. Call login() first.")
        
        # Check if model is CID or file path
        if self.ipfs.is_valid_cid(model):
            model_cid = model
            local_path = None
            print(f"   📎 Using existing IPFS CID: {model_cid}")
        else:
            if not os.path.exists(model):
                raise FileNotFoundError(f"Model file not found: {model}")
            
            print(f"   📤 Uploading model to IPFS...")
            result = self.ipfs.upload(model)
            model_cid = result.cid
            local_path = os.path.abspath(model)
            print(f"   ✓ Uploaded! CID: {model_cid}")
        
        # Validate dataset
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}. Use one of: {', '.join(POPULAR_DATASETS)}")
        
        # Get next round ID
        round_id = self.get_next_round_id()
        
        # Convert reward to lamports
        reward_lamports = int(reward * 1e9)
        
        print(f"   🔄 Creating round #{round_id}...")
        
        # Create round on blockchain
        tx = self._create_round_tx(round_id, model_cid, dataset, reward_lamports)
        
        print(f"   ✓ Created! TX: {tx[:40]}...")
        
        # Store locally
        stored = StoredRound(
            round_id=round_id,
            model_cid=model_cid,
            dataset=dataset,
            reward_amount=reward_lamports,
            created_at=datetime.now().isoformat(),
            tx_signature=tx,
            status="waitingValidator",
            local_model_path=local_path,
        )
        self.storage.add(stored)
        
        return round_id
    
    def _create_round_tx(
        self, 
        round_id: int, 
        model_cid: str, 
        dataset: str, 
        reward_lamports: int
    ) -> str:
        """Create round transaction"""
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair, SystemProgram }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const creator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(creator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const rewardAmount = new anchor.BN({reward_lamports});
    
    const [counterPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round_counter")],
        programId
    );
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [vaultPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const tx = await program.methods
        .createRound(roundId, "{model_cid}", {{ {dataset.lower()}: {{}} }}, rewardAmount)
        .accounts({{
            roundCounter: counterPda,
            round: roundPda,
            vault: vaultPda,
            creator: creator.publicKey,
            systemProgram: SystemProgram.programId,
        }})
        .signers([creator])
        .rpc();
    
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def cancel_round(self, round_id: int) -> str:
        """Cancel a round and get refund"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair, SystemProgram }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const creator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(creator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [vaultPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const tx = await program.methods
        .cancelRound(roundId)
        .accounts({{
            round: roundPda,
            vault: vaultPda,
            creator: creator.publicKey,
            systemProgram: SystemProgram.programId,
        }})
        .signers([creator])
        .rpc();
    
    console.log(tx);
}})();
'''
        tx = self._run_node(script)
        
        # Update local storage
        self.storage.update(round_id, status="cancelled")
        
        return tx
    
    def get_round_status(self, round_id: int) -> RoundInfo:
        """Get current status of a round from blockchain"""
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey }} = require("@solana/web3.js");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));

(async () => {{
    const provider = new anchor.AnchorProvider(conn, {{ publicKey: PublicKey.default }}, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const round = await program.account.round.fetch(roundPda);
    
    const statusMap = {{
        waitingValidator: "waitingValidator",
        waitingTrainers: "waitingTrainers", 
        training: "training",
        validating: "validating",
        completed: "completed",
        expired: "expired",
        cancelled: "cancelled",
    }};
    
    const status = Object.keys(round.status)[0];
    
    console.log(JSON.stringify({{
        id: round.id.toNumber(),
        creator: round.creator.toBase58(),
        modelCid: round.modelCid,
        dataset: Object.keys(round.dataset)[0],
        rewardAmount: round.rewardAmount.toNumber(),
        status: status,
        validator: round.validator.toBase58(),
        trainersCount: round.trainersCount,
        submissionsCount: round.submissionsCount,
    }}));
}})();
'''
        result = self._run_node(script)
        data = json.loads(result)
        
        # Update local storage
        self.storage.update(
            round_id,
            status=data["status"],
            validator=data["validator"] if data["validator"] != "11111111111111111111111111111111" else None,
            trainers_count=data["trainersCount"],
            submissions_count=data["submissionsCount"],
        )
        
        return RoundInfo(
            id=data["id"],
            creator=data["creator"],
            model_cid=data["modelCid"],
            dataset=data["dataset"],
            reward_amount=data["rewardAmount"],
            status=data["status"],
            validator=data["validator"],
            trainers_count=data["trainersCount"],
            submissions_count=data["submissionsCount"],
        )
    
    def list_trainings(self, refresh: bool = False) -> List[StoredRound]:
        """
        List all my training rounds.
        
        Args:
            refresh: If True, fetch latest status from blockchain
        """
        rounds = self.storage.list_all()
        
        if refresh:
            for r in rounds:
                if r.status not in ("completed", "cancelled", "expired"):
                    try:
                        self.get_round_status(r.round_id)
                    except Exception:
                        pass
            # Reload after update
            rounds = self.storage.list_all()
        
        return rounds
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        return self.storage.get_stats()
    
    def watch(self, round_id: int, interval: int = 5):
        """
        Watch round progress in real-time.
        
        Args:
            round_id: Round ID to watch
            interval: Polling interval in seconds
        """
        import time
        
        print(f"\n👁️  Watching round #{round_id}... (Ctrl+C to stop)\n")
        
        last_status = None
        
        try:
            while True:
                info = self.get_round_status(round_id)
                
                if info.status != last_status:
                    self._print_status_change(info, last_status)
                    last_status = info.status
                    
                    if info.status in ("completed", "cancelled", "expired"):
                        print("\n   Done!")
                        break
                else:
                    # Just update progress
                    print(f"\r   ⏳ {info.status}: {info.submissions_count}/{info.trainers_count} submitted   ", end="", flush=True)
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n   Stopped watching.")
    
    def _print_status_change(self, info: RoundInfo, prev_status: Optional[str]):
        """Print status change"""
        status_messages = {
            "waitingValidator": "⏳ Waiting for validator...",
            "waitingTrainers": f"✓ Validator claimed! Waiting for trainers...",
            "training": f"🏋️ Training started! {info.trainers_count} trainer(s)",
            "validating": f"🔬 Validating {info.submissions_count} submissions...",
            "completed": "✅ Training completed!",
            "expired": "⏰ Round expired",
            "cancelled": "❌ Round cancelled",
        }
        
        msg = status_messages.get(info.status, f"Status: {info.status}")
        print(f"\n   {msg}")

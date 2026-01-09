"""
Solana client for Decloud - Creator operations
create_round, finalize_round, cancel_round, force_finalize, withdraw_remainder
"""
import struct
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
import base58

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction
from solders.message import Message
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts

from config import PROGRAM_ID, TREASURY, config, DATASETS, DATASET_ID_TO_NAME


@dataclass
class RoundInfo:
    """Round information"""
    id: int
    creator: str
    model_cid: str
    dataset: str
    dataset_id: int
    reward_amount: int
    created_at: int
    status: str
    pre_count: int
    pre_accuracy_sum: int
    gradients_count: int
    total_validations: int
    total_improvement: int
    bump: int
    vault_bump: int


@dataclass
class GradientInfo:
    """Gradient information"""
    round_id: int
    trainer: str
    cid: str
    post_count: int
    post_accuracy_sum: int
    improvement: int
    reward_claimed: bool


class SolanaClient:
    """Solana client for Creator operations"""
    
    DISCRIMINATORS = {
        "initialize": bytes([175, 175, 109, 31, 13, 152, 155, 237]),
        "create_round": bytes([199, 56, 85, 38, 202, 24, 220, 227]),
        "cancel_round": bytes([218, 124, 84, 179, 14, 25, 233, 15]),
        "finalize_round": bytes([201, 41, 45, 83, 117, 52, 65, 67]),
        "force_finalize": bytes([51, 118, 143, 113, 191, 27, 69, 252]),
        "withdraw_remainder": bytes([144, 86, 175, 90, 21, 213, 206, 105]),
    }
    
    def __init__(self, keypair: Optional[Keypair] = None):
        self.program_id = Pubkey.from_string(PROGRAM_ID)
        self.treasury = Pubkey.from_string(TREASURY)
        self.client = Client(config.rpc_url)
        self.keypair = keypair
    
    @classmethod
    def from_private_key(cls, private_key: str) -> "SolanaClient":
        """Create from base58 private key"""
        secret = base58.b58decode(private_key)
        keypair = Keypair.from_bytes(secret)
        return cls(keypair)
    
    @property
    def pubkey(self) -> Optional[Pubkey]:
        return self.keypair.pubkey() if self.keypair else None
    
    def get_balance(self) -> int:
        """Get balance in lamports"""
        if not self.pubkey:
            return 0
        response = self.client.get_balance(self.pubkey, commitment=Confirmed)
        return response.value
    
    # ═══════════════════════════════════════════════════════════════
    # PDA Derivation
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_counter_pda(self) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address([b"round_counter"], self.program_id)
    
    def get_round_pda(self, round_id: int) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address(
            [b"round", round_id.to_bytes(8, "little")],
            self.program_id
        )
    
    def get_vault_pda(self, round_id: int) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address(
            [b"vault", round_id.to_bytes(8, "little")],
            self.program_id
        )
    
    def get_gradient_pda(self, round_id: int, trainer: Pubkey) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address(
            [b"gradient", round_id.to_bytes(8, "little"), bytes(trainer)],
            self.program_id
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Read Operations
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_count(self) -> int:
        """Get total rounds count"""
        pda, _ = self.get_round_counter_pda()
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return 0
        
        data = response.value.data
        count = struct.unpack("<Q", data[8:16])[0]
        return count
    
    def get_round(self, round_id: int) -> Optional[RoundInfo]:
        """Get round info"""
        pda, _ = self.get_round_pda(round_id)
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return None
        
        data = bytes(response.value.data)
        return self._parse_round(data)
    
    def _parse_round(self, data: bytes) -> RoundInfo:
        offset = 8  # discriminator
        
        id = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        creator = base58.b58encode(data[offset:offset+32]).decode()
        offset += 32
        
        model_cid_bytes = data[offset:offset+64]
        offset += 64
        model_cid_len = data[offset]
        offset += 1
        model_cid = model_cid_bytes[:model_cid_len].decode("utf-8", errors="ignore")
        
        dataset_id = data[offset]
        offset += 1
        dataset = DATASET_ID_TO_NAME.get(dataset_id, f"Unknown({dataset_id})")
        
        reward_amount = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        created_at = struct.unpack("<q", data[offset:offset+8])[0]
        offset += 8
        
        status_id = data[offset]
        offset += 1
        status_map = {0: "Active", 1: "Finalized", 2: "Cancelled"}
        status = status_map.get(status_id, "Unknown")
        
        pre_count = data[offset]
        offset += 1
        
        pre_accuracy_sum = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        gradients_count = data[offset]
        offset += 1
        
        total_validations = struct.unpack("<H", data[offset:offset+2])[0]
        offset += 2
        
        total_improvement = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        bump = data[offset]
        offset += 1
        
        vault_bump = data[offset]
        
        return RoundInfo(
            id=id, creator=creator, model_cid=model_cid, dataset=dataset,
            dataset_id=dataset_id, reward_amount=reward_amount, created_at=created_at,
            status=status, pre_count=pre_count, pre_accuracy_sum=pre_accuracy_sum,
            gradients_count=gradients_count, total_validations=total_validations,
            total_improvement=total_improvement, bump=bump, vault_bump=vault_bump,
        )
    
    def get_gradient(self, round_id: int, trainer: Pubkey) -> Optional[GradientInfo]:
        """Get gradient info"""
        pda, _ = self.get_gradient_pda(round_id, trainer)
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return None
        
        data = bytes(response.value.data)
        return self._parse_gradient(data)
    
    def _parse_gradient(self, data: bytes) -> GradientInfo:
        offset = 8
        
        round_id = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        trainer = base58.b58encode(data[offset:offset+32]).decode()
        offset += 32
        
        cid_bytes = data[offset:offset+64]
        offset += 64
        cid_len = data[offset]
        offset += 1
        cid = cid_bytes[:cid_len].decode("utf-8", errors="ignore")
        
        post_count = data[offset]
        offset += 1
        
        post_accuracy_sum = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        improvement = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        reward_claimed = bool(data[offset])
        
        return GradientInfo(
            round_id=round_id, trainer=trainer, cid=cid,
            post_count=post_count, post_accuracy_sum=post_accuracy_sum,
            improvement=improvement, reward_claimed=reward_claimed,
        )
    
    def get_my_rounds(self) -> List[RoundInfo]:
        """Get rounds created by this wallet"""
        if not self.pubkey:
            return []
        
        rounds = []
        count = self.get_round_count()
        my_pubkey = str(self.pubkey)
        
        for i in range(count):
            round_info = self.get_round(i)
            if round_info and round_info.creator == my_pubkey:
                rounds.append(round_info)
        
        return rounds
    
    def get_all_gradients(self, round_id: int) -> List[GradientInfo]:
        """Get all gradients for a round"""
        # This requires getProgramAccounts with filters
        # For now, return empty - would need proper implementation
        return []
    
    # ═══════════════════════════════════════════════════════════════
    # Write Operations
    # ═══════════════════════════════════════════════════════════════
    
    def _send_transaction(self, instruction: Instruction) -> str:
        """Send transaction with retry"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        for attempt in range(3):
            try:
                recent_blockhash = self.client.get_latest_blockhash(commitment=Confirmed).value.blockhash
                
                message = Message.new_with_blockhash(
                    [instruction],
                    self.keypair.pubkey(),
                    recent_blockhash
                )
                
                tx = Transaction.new_unsigned(message)
                tx.sign([self.keypair], recent_blockhash)
                
                response = self.client.send_transaction(
                    tx,
                    opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
                )
                return str(response.value)
            
            except Exception as e:
                if "Blockhash not found" in str(e) and attempt < 2:
                    time.sleep(1)
                    continue
                raise
        
        raise Exception("Failed after 3 attempts")
    
    def create_round(
        self,
        model_cid: str,
        dataset: str,
        reward_lamports: int,
    ) -> Tuple[str, int]:
        """
        Create a new training round
        Returns: (tx_signature, round_id)
        """
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        dataset_id = DATASETS[dataset]
        round_id = self.get_round_count()
        
        round_counter_pda, _ = self.get_round_counter_pda()
        round_pda, _ = self.get_round_pda(round_id)
        vault_pda, _ = self.get_vault_pda(round_id)
        
        # Build instruction data
        data = self.DISCRIMINATORS["create_round"]
        data += struct.pack("<Q", round_id)
        
        # String: 4 bytes length + utf8 bytes
        cid_bytes = model_cid.encode("utf-8")
        data += struct.pack("<I", len(cid_bytes))
        data += cid_bytes
        
        # Dataset enum (1 byte)
        data += bytes([dataset_id])
        
        # Reward amount
        data += struct.pack("<Q", reward_lamports)
        
        accounts = [
            AccountMeta(round_counter_pda, is_signer=False, is_writable=True),
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(vault_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        tx = self._send_transaction(instruction)
        
        return tx, round_id
    
    def cancel_round(self, round_id: int) -> str:
        """Cancel a round (only if no participants)"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        vault_pda, _ = self.get_vault_pda(round_id)
        
        data = self.DISCRIMINATORS["cancel_round"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(vault_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def finalize_round(self, round_id: int) -> str:
        """Finalize a round (creator only)"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        
        data = self.DISCRIMINATORS["finalize_round"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def force_finalize(self, round_id: int) -> str:
        """Force finalize after deadline (anyone can call)"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        
        data = self.DISCRIMINATORS["force_finalize"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def withdraw_remainder(self, round_id: int) -> str:
        """Withdraw remaining funds from finalized round"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        vault_pda, _ = self.get_vault_pda(round_id)
        
        data = self.DISCRIMINATORS["withdraw_remainder"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=False),
            AccountMeta(vault_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)

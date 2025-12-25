"""
Local Storage for Creator Rounds
================================

Stores information about created rounds locally for tracking.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class StoredRound:
    """Locally stored round information"""
    round_id: int
    model_cid: str
    dataset: str
    reward_amount: int  # lamports
    created_at: str  # ISO timestamp
    tx_signature: str
    
    # Status tracking
    status: str = "created"
    validator: Optional[str] = None
    trainers_count: int = 0
    submissions_count: int = 0
    
    # Result
    completed_at: Optional[str] = None
    result_model_cid: Optional[str] = None
    
    # Local paths
    local_model_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "StoredRound":
        return cls(**data)


class RoundStorage:
    """
    Persistent storage for created rounds.
    
    Stores in ~/.decloud-creator/rounds.json
    """
    
    def __init__(self, data_dir: str = "~/.decloud-creator"):
        self.data_dir = os.path.expanduser(data_dir)
        self.storage_path = os.path.join(self.data_dir, "rounds.json")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self._rounds: Dict[int, StoredRound] = {}
        self._load()
    
    def _load(self):
        """Load rounds from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self._rounds = {
                        int(k): StoredRound.from_dict(v) 
                        for k, v in data.items()
                    }
            except Exception as e:
                print(f"⚠️ Failed to load rounds: {e}")
                self._rounds = {}
    
    def _save(self):
        """Save rounds to disk"""
        data = {str(k): v.to_dict() for k, v in self._rounds.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add(self, round_info: StoredRound):
        """Add a new round"""
        self._rounds[round_info.round_id] = round_info
        self._save()
    
    def get(self, round_id: int) -> Optional[StoredRound]:
        """Get round by ID"""
        return self._rounds.get(round_id)
    
    def update(self, round_id: int, **kwargs):
        """Update round fields"""
        if round_id in self._rounds:
            for key, value in kwargs.items():
                if hasattr(self._rounds[round_id], key):
                    setattr(self._rounds[round_id], key, value)
            self._save()
    
    def list_all(self) -> List[StoredRound]:
        """Get all rounds sorted by creation time (newest first)"""
        return sorted(
            self._rounds.values(), 
            key=lambda r: r.created_at, 
            reverse=True
        )
    
    def list_active(self) -> List[StoredRound]:
        """Get active (non-completed) rounds"""
        return [
            r for r in self._rounds.values() 
            if r.status not in ("completed", "expired", "cancelled")
        ]
    
    def delete(self, round_id: int):
        """Delete a round"""
        if round_id in self._rounds:
            del self._rounds[round_id]
            self._save()
    
    def clear(self):
        """Clear all rounds"""
        self._rounds = {}
        self._save()
    
    def get_stats(self) -> dict:
        """Get statistics"""
        rounds = list(self._rounds.values())
        
        by_status = {}
        total_spent = 0
        
        for r in rounds:
            status = r.status
            by_status[status] = by_status.get(status, 0) + 1
            total_spent += r.reward_amount
        
        return {
            "total_rounds": len(rounds),
            "by_status": by_status,
            "total_spent_lamports": total_spent,
            "total_spent_sol": total_spent / 1e9,
        }


def format_round_status(r: StoredRound) -> str:
    """Format round for display"""
    status_icons = {
        "created": "🆕",
        "waitingValidator": "⏳",
        "waitingTrainers": "👥",
        "training": "🏋️",
        "validating": "🔬",
        "completed": "✅",
        "expired": "⏰",
        "cancelled": "❌",
    }
    
    icon = status_icons.get(r.status, "❓")
    reward_sol = r.reward_amount / 1e9
    
    line = f"{icon} Round #{r.round_id}: {r.status}"
    line += f" | {r.dataset} | {reward_sol:.4f} SOL"
    
    if r.trainers_count > 0:
        line += f" | {r.submissions_count}/{r.trainers_count} submitted"
    
    return line

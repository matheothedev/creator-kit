"""
IPFS Upload Module
==================

Upload files to IPFS via various providers.
"""

import os
import requests
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class UploadResult:
    """Result of IPFS upload"""
    cid: str
    size: int
    provider: str


class IPFSUploader:
    """
    Upload files to IPFS.
    
    Supports:
    - Pinata (recommended, free 1GB)
    - Web3.Storage (free 1TB)
    - Local IPFS node
    - NFT.Storage (free, for NFTs)
    """
    
    GATEWAYS = [
        "https://ipfs.io/ipfs/",
        "https://cloudflare-ipfs.com/ipfs/",
        "https://gateway.pinata.cloud/ipfs/",
        "https://dweb.link/ipfs/",
        "https://w3s.link/ipfs/",
    ]
    
    def __init__(
        self,
        pinata_api_key: Optional[str] = None,
        pinata_secret_key: Optional[str] = None,
        web3_storage_token: Optional[str] = None,
        local_ipfs_url: Optional[str] = None,
    ):
        self.pinata_api_key = pinata_api_key or os.getenv("PINATA_API_KEY")
        self.pinata_secret_key = pinata_secret_key or os.getenv("PINATA_SECRET_KEY")
        self.web3_storage_token = web3_storage_token or os.getenv("WEB3_STORAGE_TOKEN")
        self.local_ipfs_url = local_ipfs_url or os.getenv("IPFS_API_URL", "http://localhost:5001")
    
    def upload(self, file_path: str) -> UploadResult:
        """
        Upload file to IPFS using available provider.
        
        Priority:
        1. Pinata (if configured)
        2. Web3.Storage (if configured)
        3. Local IPFS node
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        
        # Try Pinata first
        if self.pinata_api_key and self.pinata_secret_key:
            try:
                cid = self._upload_pinata(file_path)
                return UploadResult(cid=cid, size=file_size, provider="pinata")
            except Exception as e:
                print(f"   ⚠️ Pinata failed: {e}")
        
        # Try Web3.Storage
        if self.web3_storage_token:
            try:
                cid = self._upload_web3storage(file_path)
                return UploadResult(cid=cid, size=file_size, provider="web3.storage")
            except Exception as e:
                print(f"   ⚠️ Web3.Storage failed: {e}")
        
        # Try local IPFS
        try:
            cid = self._upload_local(file_path)
            return UploadResult(cid=cid, size=file_size, provider="local")
        except Exception as e:
            print(f"   ⚠️ Local IPFS failed: {e}")
        
        raise RuntimeError(
            "No IPFS provider available!\n"
            "Configure one of:\n"
            "  - Pinata: export PINATA_API_KEY=xxx PINATA_SECRET_KEY=yyy\n"
            "  - Web3.Storage: export WEB3_STORAGE_TOKEN=xxx\n"
            "  - Local: ipfs daemon"
        )
    
    def _upload_pinata(self, file_path: str) -> str:
        """Upload to Pinata"""
        url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
        
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                url,
                files={'file': (filename, f)},
                headers={
                    'pinata_api_key': self.pinata_api_key,
                    'pinata_secret_api_key': self.pinata_secret_key
                },
                timeout=300
            )
        
        if response.status_code != 200:
            raise RuntimeError(f"Pinata error: {response.text}")
        
        return response.json()['IpfsHash']
    
    def _upload_web3storage(self, file_path: str) -> str:
        """Upload to Web3.Storage"""
        url = "https://api.web3.storage/upload"
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                url,
                files={'file': f},
                headers={
                    'Authorization': f'Bearer {self.web3_storage_token}'
                },
                timeout=300
            )
        
        if response.status_code != 200:
            raise RuntimeError(f"Web3.Storage error: {response.text}")
        
        return response.json()['cid']
    
    def _upload_local(self, file_path: str) -> str:
        """Upload to local IPFS node"""
        url = f"{self.local_ipfs_url}/api/v0/add"
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                url,
                files={'file': f},
                timeout=300
            )
        
        if response.status_code != 200:
            raise RuntimeError(f"Local IPFS error: {response.text}")
        
        return response.json()['Hash']
    
    def download(self, cid: str, output_path: str, timeout: int = 120) -> bool:
        """Download file from IPFS"""
        for gateway in self.GATEWAYS:
            try:
                url = f"{gateway}{cid}"
                response = requests.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    return True
            except Exception:
                continue
        
        return False
    
    def is_valid_cid(self, cid: str) -> bool:
        """Check if CID is valid format"""
        # Basic validation: CIDv0 starts with Qm (46 chars), CIDv1 starts with ba (59 chars)
        if len(cid) < 46 or len(cid) > 64:
            return False
        
        if cid.startswith('Qm') or cid.startswith('ba'):
            return True
        
        return False
    
    def get_url(self, cid: str) -> str:
        """Get public URL for CID"""
        return f"https://ipfs.io/ipfs/{cid}"


def setup_pinata_interactive() -> Tuple[str, str]:
    """Interactive Pinata setup"""
    print("\n📌 Pinata Setup")
    print("   Get free API keys at: https://app.pinata.cloud/keys")
    print()
    
    api_key = input("   API Key: ").strip()
    secret_key = input("   Secret Key: ").strip()
    
    # Test connection
    try:
        response = requests.get(
            "https://api.pinata.cloud/data/testAuthentication",
            headers={
                'pinata_api_key': api_key,
                'pinata_secret_api_key': secret_key
            }
        )
        if response.status_code == 200:
            print("   ✓ Pinata connected!")
            return api_key, secret_key
        else:
            print(f"   ❌ Authentication failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
    
    return "", ""

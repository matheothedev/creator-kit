"""
IPFS client for downloading model packages and gradients
"""
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console

from config import IPFS_GATEWAYS, CACHE_DIR

console = Console()


class IPFSClient:
    """Client for downloading from IPFS"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.gateways = IPFS_GATEWAYS
        
    async def download_file(self, cid: str, filename: str) -> Optional[bytes]:
        """Download single file from IPFS"""
        for gateway in self.gateways:
            try:
                url = f"{gateway}{cid}/{filename}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 200:
                            return await resp.read()
            except Exception as e:
                continue
        return None
    
    async def download_package(self, cid: str) -> Optional[Path]:
        """Download model/gradient package from IPFS"""
        cache_path = self.cache_dir / cid
        
        # Check cache
        if cache_path.exists() and (cache_path / "config.json").exists():
            return cache_path
        
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Required files
        files = ["config.json", "head.safetensors", "embeddings.safetensors"]
        
        for filename in files:
            content = await self.download_file(cid, filename)
            if content:
                with open(cache_path / filename, "wb") as f:
                    f.write(content)
            elif filename == "config.json":
                # config.json is required
                console.print(f"[red]Failed to download {filename}[/red]")
                return None
        
        return cache_path
    
    async def download_gradient(self, cid: str) -> Optional[Path]:
        """Download gradient package (only head + config)"""
        cache_path = self.cache_dir / f"gradient_{cid}"
        
        if cache_path.exists() and (cache_path / "config.json").exists():
            return cache_path
        
        cache_path.mkdir(parents=True, exist_ok=True)
        
        files = ["config.json", "head.safetensors"]
        
        for filename in files:
            content = await self.download_file(cid, filename)
            if content:
                with open(cache_path / filename, "wb") as f:
                    f.write(content)
            elif filename in ["config.json", "head.safetensors"]:
                return None
        
        return cache_path
    
    def download_package_sync(self, cid: str) -> Optional[Path]:
        """Sync wrapper"""
        return asyncio.run(self.download_package(cid))
    
    def download_gradient_sync(self, cid: str) -> Optional[Path]:
        """Sync wrapper"""
        return asyncio.run(self.download_gradient(cid))


ipfs_client = IPFSClient()

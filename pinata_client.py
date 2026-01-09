"""
Pinata client for uploading to IPFS
"""
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import Optional
from rich.console import Console

from config import config

console = Console()

PINATA_API_URL = "https://api.pinata.cloud"


class PinataClient:
    """Client for Pinata IPFS uploads"""
    
    def __init__(self):
        pass
    
    def _get_headers(self) -> dict:
        """Get auth headers"""
        if config.pinata_jwt:
            return {"Authorization": f"Bearer {config.pinata_jwt}"}
        elif config.pinata_api_key and config.pinata_secret_key:
            return {
                "pinata_api_key": config.pinata_api_key,
                "pinata_secret_key": config.pinata_secret_key,
            }
        return {}
    
    def test_authentication_sync(self) -> bool:
        """Test Pinata auth"""
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{PINATA_API_URL}/data/testAuthentication",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    async def test_authentication(self) -> bool:
        """Async test auth"""
        try:
            headers = self._get_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{PINATA_API_URL}/data/testAuthentication",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def upload_directory(self, directory: Path, name: str = "decloud_package") -> Optional[str]:
        """Upload directory to Pinata, returns CID"""
        headers = self._get_headers()
        
        if not headers:
            console.print("[red]Pinata not configured[/red]")
            return None
        
        try:
            # Prepare multipart form
            data = aiohttp.FormData()
            
            # Add all files
            for file_path in directory.iterdir():
                if file_path.is_file():
                    data.add_field(
                        'file',
                        open(file_path, 'rb'),
                        filename=f"{name}/{file_path.name}",
                        content_type='application/octet-stream'
                    )
            
            # Pinata options
            data.add_field(
                'pinataOptions',
                '{"cidVersion": 1}',
                content_type='application/json'
            )
            
            data.add_field(
                'pinataMetadata',
                f'{{"name": "{name}"}}',
                content_type='application/json'
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{PINATA_API_URL}/pinning/pinFileToIPFS",
                    headers=headers,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("IpfsHash")
                    else:
                        text = await resp.text()
                        console.print(f"[red]Pinata error: {resp.status} - {text}[/red]")
                        return None
                        
        except Exception as e:
            console.print(f"[red]Upload error: {e}[/red]")
            return None
    
    def upload_directory_sync(self, directory: Path, name: str = "decloud_package") -> Optional[str]:
        """Sync upload"""
        return asyncio.run(self.upload_directory(directory, name))
    
    async def upload_model_package(self, package_dir: Path, dataset: str) -> Optional[str]:
        """Upload model package with metadata"""
        name = f"decloud_model_{dataset}"
        return await self.upload_directory(package_dir, name)
    
    def upload_model_package_sync(self, package_dir: Path, dataset: str) -> Optional[str]:
        """Sync wrapper"""
        return asyncio.run(self.upload_model_package(package_dir, dataset))


pinata_client = PinataClient()

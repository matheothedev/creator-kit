"""
Lighthouse Storage client for IPFS uploads
Uses Solana keypair for authentication
"""
import asyncio
import aiohttp
import requests
import json
import time
from pathlib import Path
from typing import Optional, Tuple
import base58
from nacl.signing import SigningKey
from nacl.encoding import RawEncoder
from rich.console import Console

console = Console()

LIGHTHOUSE_API_URL = "https://api.lighthouse.storage"
LIGHTHOUSE_NODE_URL = "https://upload.lighthouse.storage"
LIGHTHOUSE_GATEWAY = "https://gateway.lighthouse.storage/ipfs/"


class LighthouseClient:
    """Client for Lighthouse Storage (IPFS) uploads using Solana keypair"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    # ═══════════════════════════════════════════════════════════════
    # Solana Keypair Utilities
    # ═══════════════════════════════════════════════════════════════
    
    @staticmethod
    def keypair_from_base58(private_key_base58: str) -> Tuple[str, SigningKey]:
        """Load keypair from base58 private key string
        
        Args:
            private_key_base58: Base58 encoded private key (64 bytes)
            
        Returns:
            tuple: (public_key_base58, signing_key)
        """
        secret_key = base58.b58decode(private_key_base58)
        # First 32 bytes are the private key seed
        signing_key = SigningKey(secret_key[:32])
        verify_key = signing_key.verify_key
        public_key_base58 = base58.b58encode(bytes(verify_key)).decode('utf-8')
        return public_key_base58, signing_key
    
    @staticmethod
    def sign_message(message: str, signing_key: SigningKey) -> str:
        """Sign a message with Solana keypair
        
        Args:
            message: Message to sign
            signing_key: NaCl signing key
            
        Returns:
            Base58 encoded signature
        """
        message_bytes = message.encode('utf-8')
        signed = signing_key.sign(message_bytes, encoder=RawEncoder)
        signature = signed.signature
        return base58.b58encode(signature).decode('utf-8')
    
    # ═══════════════════════════════════════════════════════════════
    # Authentication
    # ═══════════════════════════════════════════════════════════════
    
    @staticmethod
    def get_auth_message(public_key: str) -> str:
        """Get authentication message from Lighthouse API
        
        Args:
            public_key: Solana public key (base58)
            
        Returns:
            Message to sign
        """
        url = f"{LIGHTHOUSE_API_URL}/api/auth/get_message?publicKey={public_key}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        text = response.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text.strip('"')
    
    @staticmethod
    def create_api_key(public_key: str, signed_message: str, key_name: str = "decloud-creator") -> str:
        """Create Lighthouse API key
        
        Args:
            public_key: Solana public key (base58)
            signed_message: Signed auth message (base58)
            key_name: Name for the API key
            
        Returns:
            API key string
        """
        url = f"{LIGHTHOUSE_API_URL}/api/auth/create_api_key"
        payload = {
            "publicKey": public_key,
            "signedMessage": signed_message,
            "keyName": key_name
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        text = response.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text.strip('"')
    
    @classmethod
    def create_api_key_from_private_key(cls, private_key_base58: str, key_name: str = "decloud-creator") -> str:
        """Full flow: create Lighthouse API key from Solana private key
        
        Args:
            private_key_base58: Base58 encoded Solana private key
            key_name: Name for the API key
            
        Returns:
            API key string
        """
        # Get keypair
        public_key, signing_key = cls.keypair_from_base58(private_key_base58)
        
        console.print(f"[dim]Creating Lighthouse API key for {public_key[:20]}...[/dim]")
        
        # Get auth message
        auth_message = cls.get_auth_message(public_key)
        
        # Sign message
        signed_message = cls.sign_message(auth_message, signing_key)
        
        # Create API key
        api_key = cls.create_api_key(public_key, signed_message, key_name)
        
        return api_key
    
    # ═══════════════════════════════════════════════════════════════
    # Upload Operations
    # ═══════════════════════════════════════════════════════════════
    
    def _get_headers(self) -> dict:
        """Get auth headers"""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}
    
    def test_authentication_sync(self) -> bool:
        """Test Lighthouse API key"""
        if not self.api_key:
            return False
        
        try:
            # Try to get user data usage as auth test
            headers = self._get_headers()
            response = requests.get(
                f"{LIGHTHOUSE_API_URL}/api/user/user_data_usage",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    async def test_authentication(self) -> bool:
        """Async test auth"""
        if not self.api_key:
            return False
        
        try:
            headers = self._get_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{LIGHTHOUSE_API_URL}/api/user/user_data_usage",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    return resp.status == 200
        except:
            return False
    
    async def upload_file(self, file_path: Path, name: Optional[str] = None) -> Optional[str]:
        """Upload single file to Lighthouse
        
        Args:
            file_path: Path to file
            name: Optional name for the file
            
        Returns:
            CID if successful
        """
        if not self.api_key:
            console.print("[red]Lighthouse API key not configured[/red]")
            return None
        
        try:
            headers = self._get_headers()
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field(
                    'file',
                    open(file_path, 'rb'),
                    filename=name or file_path.name,
                    content_type='application/octet-stream'
                )
                
                async with session.post(
                    f"{LIGHTHOUSE_NODE_URL}/api/v0/add",
                    headers=headers,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get('Hash')
                    else:
                        text = await resp.text()
                        console.print(f"[red]Lighthouse upload error: {resp.status} - {text}[/red]")
                        return None
                        
        except Exception as e:
            console.print(f"[red]Upload error: {e}[/red]")
            return None
    
    async def upload_directory(self, directory: Path, name: str = "decloud_package") -> Optional[str]:
        """Upload directory to Lighthouse, returns CID
        
        Args:
            directory: Path to directory
            name: Name for the package
            
        Returns:
            CID if successful
        """
        if not self.api_key:
            console.print("[red]Lighthouse API key not configured[/red]")
            return None
        
        try:
            headers = self._get_headers()
            
            # Collect all files
            files_to_upload = []
            for file_path in directory.iterdir():
                if file_path.is_file():
                    files_to_upload.append(file_path)
            
            if not files_to_upload:
                console.print("[red]No files to upload[/red]")
                return None
            
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                
                # Add all files with directory prefix
                for file_path in files_to_upload:
                    data.add_field(
                        'file',
                        open(file_path, 'rb'),
                        filename=f"{name}/{file_path.name}",
                        content_type='application/octet-stream'
                    )
                
                async with session.post(
                    f"{LIGHTHOUSE_NODE_URL}/api/v0/add",
                    headers=headers,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=600)
                ) as resp:
                    if resp.status == 200:
                        # Response may be multiple JSON objects (one per file + directory)
                        text = await resp.text()
                        lines = text.strip().split('\n')
                        
                        # Find the directory entry (it's the last one with the folder name)
                        for line in reversed(lines):
                            try:
                                result = json.loads(line)
                                if result.get('Name') == name:
                                    return result.get('Hash')
                            except:
                                continue
                        
                        # Fallback: return last hash
                        if lines:
                            try:
                                result = json.loads(lines[-1])
                                return result.get('Hash')
                            except:
                                pass
                        
                        return None
                    else:
                        text = await resp.text()
                        console.print(f"[red]Lighthouse upload error: {resp.status} - {text}[/red]")
                        return None
                        
        except Exception as e:
            console.print(f"[red]Upload error: {e}[/red]")
            return None
    
    def upload_directory_sync(self, directory: Path, name: str = "decloud_package") -> Optional[str]:
        """Sync upload directory"""
        return asyncio.run(self.upload_directory(directory, name))
    
    async def upload_model_package(self, package_dir: Path, dataset: str) -> Optional[str]:
        """Upload model package with metadata
        
        Args:
            package_dir: Path to package directory
            dataset: Dataset name
            
        Returns:
            CID if successful
        """
        name = f"decloud_model_{dataset}"
        return await self.upload_directory(package_dir, name)
    
    def upload_model_package_sync(self, package_dir: Path, dataset: str) -> Optional[str]:
        """Sync wrapper"""
        return asyncio.run(self.upload_model_package(package_dir, dataset))
    
    # ═══════════════════════════════════════════════════════════════
    # Account Info
    # ═══════════════════════════════════════════════════════════════
    
    def get_data_usage(self) -> Optional[dict]:
        """Get account data usage"""
        if not self.api_key:
            return None
        
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{LIGHTHOUSE_API_URL}/api/user/user_data_usage",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_uploads(self) -> Optional[list]:
        """Get list of uploaded files"""
        if not self.api_key:
            return None
        
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{LIGHTHOUSE_API_URL}/api/user/files_uploaded",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('fileList', [])
        except:
            pass
        return None


# Singleton instance (will be initialized with API key from config)
lighthouse_client: Optional[LighthouseClient] = None


def init_lighthouse_client(api_key: Optional[str] = None):
    """Initialize the global lighthouse client"""
    global lighthouse_client
    lighthouse_client = LighthouseClient(api_key)
    return lighthouse_client


def get_lighthouse_client() -> LighthouseClient:
    """Get the global lighthouse client"""
    global lighthouse_client
    if lighthouse_client is None:
        lighthouse_client = LighthouseClient()
    return lighthouse_client

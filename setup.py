from setuptools import setup, find_packages

setup(
    name="decloud-creator",
    version="1.0.0",
    description="Decloud Creator Kit - Create training rounds on Solana",
    author="Decloud",
    py_modules=[
        "config",
        "solana_client",
        "ipfs_client",
        "lighthouse_client",
        "model_builder",
        "creator",
        "main",
    ],
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "solana>=0.30.0",
        "solders>=0.18.0",
        "base58>=2.1.0",
        "pynacl>=1.5.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
    ],
    entry_points={
        "console_scripts": [
            "decloud-creator=main:main",
        ],
    },
    python_requires=">=3.9",
)

from setuptools import setup, find_packages

setup(
    name="decloud-creator-kit",
    version="0.1.0",
    description="DECLOUD Creator Kit - Create federated learning rounds",
    author="DECLOUD Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
    ],
    entry_points={
        "console_scripts": [
            "decloud-creator=decloud_creator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["idl.json"],
    },
)

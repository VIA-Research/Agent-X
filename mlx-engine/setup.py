from setuptools import find_namespace_packages, setup

setup(
    name="mlx-engine",
    version="0.1.0+agentx",
    packages=find_namespace_packages(
        include=["mlx_engine", "mlx_engine.*"], exclude=["*.__pycache__"]
    ),
    python_requires=">=3.10",
)

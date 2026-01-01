"""
Setup script for Driver Drowsiness Detection backend.
"""

from setuptools import setup, find_packages

setup(
    name="drowsiness-detection",
    version="0.1.0",
    description="Real-Time Driver Drowsiness Detection System",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.1",
        "mediapipe>=0.10.8",
        "tensorflow>=2.15.0",
        "scikit-learn>=1.3.2",
        "numpy>=1.24.3",
        "pandas>=2.1.4",
        "pillow>=10.1.0",
        "scipy>=1.11.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "hypothesis>=6.92.1",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
    },
)

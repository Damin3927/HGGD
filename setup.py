from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hggd",
    version="0.1.0",
    author="THU-VCLab",
    description="Efficient Heatmap-Guided 6-Dof Grasp Detection in Cluttered Scenes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/THU-VCLab/HGGD",
    packages=find_packages(include=["hggd", "hggd.*", "customgraspnetAPI", "customgraspnetAPI.*", 
                                   "dataset", "dataset.*", "models", "models.*"]),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.5",
        "pandas",
        "numba",
        "matplotlib",
        "open3d",
        "opencv-python",
        "scikit-image",
        "tensorboardX",
        "torchsummary",
        "tqdm",
        "transforms3d",
        "trimesh",
        "autolab_core",
        "cvxopt",
    ],
    # These dependencies require special installation
    extras_require={
        "full": [
            "torch>=1.10.0",
            "pytorch3d",
            "cupoch",
            "grasp_nms",
        ]
    },
)

from setuptools import setup, find_packages

setup(
    name="convex_optimization",
    version="0.1.0",
    packages=find_packages(include=['deblur_denoise', 'deblur_denoise.*']),
    install_requires=[
        "matplotlib==3.10.0",
        "numpy==1.26.4",
        "scipy==1.14.0",
        "torch==2.0.1",
        "torchaudio==2.0.2",
        "torchvision==0.15.2",
    ],
    author="Alexander Kakabadze, Edmand Yu, Lilian Yuan, Rahul Padmanabhan",
    author_email="rahul.padmanabhan@mail.mcgill.ca",
    description="A collection of convex optimization algorithms",
    long_description=open("Project.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rahul3/convex_optimization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 
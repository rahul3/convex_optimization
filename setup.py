from setuptools import setup, find_packages

setup(
    name="convex_optimization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    author="Rahul Padmanabhan, Lilian Yuan, Alexander Kakabadze, Edmand Yu",
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
from setuptools import setup, find_packages

setup(
    name="lqrax", 
    version="0.0.6", 
    author="Max Muchen Sun",
    author_email="msun@u.northwestern.edu",
    description="JAX-enabled continuous-time Riccati equation solver", 
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maxmsun/lqrax",  
    packages=find_packages(),  
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', 
)
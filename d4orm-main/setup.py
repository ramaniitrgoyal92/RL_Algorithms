from setuptools import setup, find_packages

setup(name='d4orm',
    packages=find_packages(include="d4orm"),
    version='0.0.1',
    install_requires=[
        'gym', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'imageio',
        'control', 
        'tqdm', 
        'tyro', 
        'meshcat', 
        'sympy', 
        'gymnax',
        'jax', 
        'distrax', 
        'gputil', 
        'jaxopt'
        ]
)
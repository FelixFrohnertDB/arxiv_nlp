from setuptools import setup, find_packages


version = 0.1

install_requires = [
    "numpy",
    "scikit-learn",
    "tqdm",
    "gensim",
    "setuptools",
    "spacey",
    "nltk",
    "pandas",
    "networkx",
    "nodevectors"
]

info = {
    "name": "arxivML",
    "version": version,
    "author": "Felix Frohnert",
    "description": "A package for machine learning for arxiv data",
    "long_description": open('README.md').read(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/FelixFrohnertDB/arxiv_nlp",
    "license": "Apache 2.0",
    "provides": ["arxivML"],
    "install_requires": install_requires,
    "packages": find_packages(where='src'),
    "package_dir": {'': 'src'},
    "keywords": ["arxiv", "Machine Learning"],
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
]

setup(classifiers=classifiers, **info)

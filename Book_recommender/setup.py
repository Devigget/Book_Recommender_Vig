from setuptools import setup, find_packages

setup(
    name='book_recommender',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'numpy',
        'pandas',
        'scikit-learn',
        'seaborn',
        'scipy',
        'matplotlib',
    ],
)

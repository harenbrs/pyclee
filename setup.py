import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='pyclee',
    version='0.1',
    author='Sebastian Harenbrock',
    author_email='harenbrs@tcd.ie',
    description='A Python implementation of the DyClee dynamic clustering algorithm by Roa et al.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/harenbrs/pyclee',
    packages=['pyclee'],
    install_requires=['numpy'],
    extras_require={
        'all': [
            'matplotlib',
            'Ordered-set-37',
            'Rtree',
            'seaborn',
            'scikit-learn',
            'tqdm'
        ],
        'plotting': ['matplotlib', 'seaborn'],
        'recommended': ['Ordered-set-37', 'Rtree', 'scikit-learn', 'tqdm']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.8',
)

from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import product
from PyBioMed import Pyprotein
from PyBioMed.PyProtein import AAIndex
from joblib import Memory
from tqdm import tqdm
from numpy.fft import fft

memory = Memory(location='__feature_cache__')
memory = Memory(None)

DEBUG = False
ALPHABET = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]


def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class DummyEstimator(BaseEstimator):
    def fit(self, *args, **kwargs): pass
    def transform(self, *args, **kwargs): pass
    def score(self, *args, **kwargs): pass


class BaseTransformer(TransformerMixin):
    def fit(self, *args, **kwargs):
        return self


class OneHotEncoder(BaseEstimator, BaseTransformer):
    def transform(self, X, y=None):
        def string_vectorizer(s):
            vectors = [
                [1 if char == letter else 0 for char in ALPHABET]
                for letter in s
            ]
            flattened = [flag for vector in vectors for flag in vector]
            return pd.Series(flattened)
        variants = X['Variants']
        features = variants.apply(string_vectorizer)
        debug('OneHotEncoder created {} features'.format(features.shape[1]))
        return features


class OneHotPairEncoder(BaseEstimator, BaseTransformer):
    def transform(self, X, y=None):
        variants = X['Variants']
        pairs = list(product(ALPHABET, ALPHABET))
        def string_vectorizer(s):
            vector = []
            for letter_a, letter_b in pairs:
                if letter_a != letter_b:
                    flag = letter_a in s and letter_b in s
                else:
                    # if they're the same, check if positions are different
                    idxs = [i for i, l in enumerate(s) if l == letter_a]
                    flag = len(idxs) > 1
                vector.append(flag)
            return pd.Series(vector)
        features = variants.apply(string_vectorizer)
        debug('OneHotPairEncoder created {} features'.format(features.shape[1]))
        return features


def encode_protein_pybiomed(variant):
    print('Encoding PyBioMed:', variant)
    protein_class = Pyprotein.PyProtein(variant)
    features = {}
    attr_names = [
        'GetAAComp',
        'GetDPComp',
        'GetTPComp',
        'GetMoreauBrotoAuto',
        'GetMoranAuto',
        'GetGearyAuto',
        'GetCTD',
        'GetSOCN',
        'GetQSO',
        'GetTriad'
    ]
    for attr_name in attr_names:
        attr_val = getattr(protein_class, attr_name)
        print(attr_name)
        try:
            result = attr_val()
        except Exception as exc:
            print('Exception:', exc)
            continue
        features.update(result)
    return pd.Series(features)


def get_index_names():
    AAIndex.init()
    aaindex = AAIndex._aaindex
    aaindex1_keys = [
        key for key, val in aaindex.items()
        if not isinstance(val, AAIndex.MatrixRecord)
    ]
    return aaindex1_keys

def get_index_values(index_name):
    """Return a dict of {Letter: Float}"""
    return AAIndex.GetAAIndex1(index_name)

# TODO: use aaindex 2 and 3?
'''
aaindex2:
https://proteinstructures.com/Sequence/Sequence/amino-acid-substitution.html
In the resulting mutation data (or probability) matrix Mij each element
provides an estimate of the probability of an amino acid in column i to be
mutated to the amino acid in row j after certain evolutionary time.

aaindex3:
https://en.wikipedia.org/wiki/Statistical_potential
For pairwise amino acid contacts, a statistical potential is formulated as
an interaction matrix that assigns a weight or energy value to each
possible pair of standard amino acids. The energy of a particular
structural model is then the combined energy of all pairwise contacts
(defined as two amino acids within a certain distance of each other) in
the structure.
'''

def encode_protein_aaindex(variant, index_name):
    debug('Encoding aaindex:', variant)
    index_values = get_index_values(index_name)
    encoded = [index_values[letter] for letter in variant]
    rval = pd.Series(encoded)
    return rval

def encode_proteins_aaindex(variants, index_name):
    from functools import partial
    _encode_protein_aaindex = partial(
        encode_protein_aaindex,
        index_name=index_name
    )
    rval = variants.apply(_encode_protein_aaindex)
    return rval

encode_proteins_aaindex__cached = memory.cache(encode_proteins_aaindex)


def encode_proteins_pybiomed(variants):
    rval = variants.apply(encode_protein_pybiomed)
    return rval

encode_proteins_pybiomed__cached = memory.cache(encode_proteins_pybiomed)


class PyBioMedEncoder(BaseEstimator, BaseTransformer):
    def transform(self, X, y=None):
        variants = X['Variants']
        return encode_proteins_pybiomed__cached(variants)

class AAIndexEncoder(BaseEstimator, BaseTransformer):

    def __init__(self, index_name=None):
        self.index_name = index_name

    def transform(self, X, y=None):
        variants = X['Variants']
        rval = encode_proteins_aaindex__cached(variants, self.index_name)
        print('AAIndexEncoder rval.shape:', rval.shape)
        return rval


class FFTEncoder(BaseEstimator, BaseTransformer):
    """
    The output of this encoder gives the equivalent of "dataset A" in [1].

    [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6233173/#MOESM1
    """
    def transform(self, X, y=None):
        spect = fft(X)
        rows, cols = spect.shape
        rval = np.stack((spect.real, spect.imag), -1).reshape((rows, -1))
        print('FFTEncoder rval.shape:', rval.shape)
        return rval

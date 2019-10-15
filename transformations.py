from __future__ import print_function

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from itertools import product
from PyBioMed import Pyprotein
from PyBioMed.PyProtein import AAIndex
from joblib import Memory
from tqdm import tqdm

memory = Memory(location='__feature_cache__')

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


def get_aaindex():
    AAIndex.init()
    aaindex = AAIndex._aaindex
    aaindex1_keys = [
        key for key, val in aaindex.items()
        if not isinstance(val, AAIndex.MatrixRecord)
    ]
    aaindex1_features = []
    for key in tqdm(aaindex1_keys):
        aaindex1 = AAIndex.GetAAIndex1(key)
        aaindex1_features.append(aaindex1)

    # TODO: use aaindex 2 and 3
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

    return pd.DataFrame(aaindex1_features)

get_aaindex__cached = memory.cache(get_aaindex)


def encode_protein_aaindex(variant):
    print('Encoding aaindex:', variant)
    aaindex = get_aaindex__cached()
    cols = []
    for letter in variant:
        col = aaindex[letter]
        cols.append(col)
    debug('encode_protein_aaindex() cols:', cols)
    return pd.Series(pd.DataFrame(cols).values.flatten())

def encode_proteins_aaindex(variants):
    rval = variants.apply(encode_protein_aaindex)
    return rval

def encode_proteins_pybiomed(variants):
    rval = variants.apply(encode_protein_pybiomed)
    return rval


encode_proteins_pybiomed__cached = memory.cache(encode_proteins_pybiomed)
encode_proteins_aaindex__cached = memory.cache(encode_proteins_aaindex)


class PyBioMedEncoder(BaseEstimator, BaseTransformer):
    def transform(self, X, y=None):
        variants = X['Variants']
        return encode_proteins_pybiomed__cached(variants)

class AAIndexEncoder(BaseEstimator, BaseTransformer):
    def transform(self, X, y=None):
        variants = X['Variants']
        return encode_proteins_aaindex__cached(variants)

import numpy as np
from numpy.fft import fft

class FFTEncoder(BaseEstimator, BaseTransformer):
    """
    The output of this encoder gives the equivalent of "dataset A" in [1].

    [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6233173/#MOESM1
    """
    def transform(self, X, y=None):
        spect = fft(X)
        return spect.real
        rows, cols = spect.shape
        return np.stack((spect.real, spect.imag), -1).reshape((rows, -1))

from __future__ import print_function
from pprint import pprint
from itertools import product
from timeit import default_timer as timer

from joblib import Memory
from sklearn import ensemble, linear_model, neural_network, svm, tree, kernel_ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    LeaveOneOut
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from transformations import (
    ALPHABET,
    DummyEstimator,
    FFTEncoder,
    OneHotEncoder,
    OneHotPairEncoder,
    #PyBioMedEncoder,
    AAIndexEncoder,
    get_index_names
)


EXPECTED_VARIANT_LENGTH = 4
FPATH_DATASET = 'DataSet for Assignment.xlsx - Sheet1 (1).csv'
CACHEDIR = '__cache__'
NUM_FOLDS = 2  # if None, NUM_FOLDS == num_rows ==> leave one out
N_FEATURES_RATIOS = [
    .01,
    .1,
    1
]


memory = Memory(location=CACHEDIR)


def main():
    """Main"""

    search, estimator = innovSAR()
    pick_candidates(estimator, 10)


def innovSAR():
    df = load_data()
    Y = df['Fitness']
    X = df[['Variants']]
    index_names = get_index_names()
    pipeline = Pipeline(
        [
            ('encode', AAIndexEncoder()),
            ('fft', FFTEncoder()),
            ('regress', PLSRegression())
        ],
        memory
    )
    print(len(index_names), 'indexes')
    grid = {
        'encode__index_name': index_names,
        # TODO: search over hyperparameters
        'regress': [
            PLSRegression()
        ]
    }
    kfold = KFold(n_splits=NUM_FOLDS, random_state=0)
    search = GridSearchCV(
        pipeline,
        grid,
        error_score=np.nan,
        verbose=5,
        n_jobs=-1,
        cv=kfold
    )

    print('*' * 40)
    print('Searching')
    print('*' * 40)
    start = timer()
    search.fit(X, Y)
    end = timer()
    print('Finished in: {}'.format(end - start))

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    best_index = search.best_index_
    best_std = search.cv_results_['std_test_score'][best_index]
    print('best_estimator:', best_estimator)
    print('best_params:', best_params)
    print('best_score:', best_score)
    print('best_std:', best_std)

    return search, best_estimator


def pick_candidates(estimator, num_candidates):
    df = load_data()
    all_candidates = set(product(*[ALPHABET] * EXPECTED_VARIANT_LENGTH))
    existing_candidates = set(df['Variants'])
    new_candidates = all_candidates - existing_candidates
    score_candidate_tups = []
    print('Scoring candidates...')
    for candidate in tqdm(new_candidates):
        prediction = estimator.predict(pd.DataFrame({'Variants': [candidate]}))
        score_candidate_tups.append((prediction, candidate))
    score_candidate_tups.sort(key=lambda tup: tup[0])
    top_candidates = score_candidate_tups[:num_candidates]
    print('Top candidates:')
    pprint(top_candidates)
        

def load_data():
    """Load data and run some sanity checks"""

    df = pd.read_csv(FPATH_DATASET)
    df.append({'Variants': 'VDGV', 'Fitness': 1}, ignore_index=True)
    variants = df['Variants']
    unique_variants = variants.unique()
    assert len(unique_variants) == len(variants)
    alphabet = set()
    lengths = set()
    variants.apply(lambda variant: alphabet.update(set(variant)))
    alphabet = sorted(list(alphabet))
    assert alphabet == ALPHABET, (alphabet, ALPHABET)
    variants.apply(lambda variant: lengths.add(len(variant)))
    assert len(lengths) == 1
    variant_length = list(lengths)[0]
    assert variant_length == EXPECTED_VARIANT_LENGTH
    return df


def get_combined_grids(grid_steps):
    combined_grids = []
    grid_step_combinations = product(*grid_steps)
    for grid_step_combination in grid_step_combinations:
        combined_grid = {}
        for grid_step in grid_step_combination:
            # don't overwrite steps
            if any([k in combined_grid for k in grid_step.keys()]):
                continue
            combined_grid.update(grid_step)
        combined_grids.append(combined_grid)
    return combined_grids


def get_best_estimator():
    """Hyperparameter optimization"""

    df = load_data()
    Y = df['Fitness']
    X = df[['Variants']]
    features = FeatureUnion([
        #('one_hot_encoder', OneHotEncoder()),
        #('one_hot_pair_encoder', OneHotPairEncoder()),
        #('pybiomed_encoder', PyBioMedEncoder()),
        ('aaindex_encoder', AAIndexEncoder())
    ])
    print('*' * 40)
    print('Extracting features...')
    print('*' * 40)
    start = timer()
    X = features.transform(X)
    end = timer()
    print('Finished in: {}'.format(end - start))
    num_rows, num_cols = X.shape
    assert num_rows == len(df)
    print('Got {} features'.format(num_cols))

    # TODO include this in pipeilne

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)
    assert not pd.DataFrame(X).isna().any().any()

    X = FFTEncoder().fit_transform(X)

    import ipdb; ipdb.set_trace()

    n_features_options = [int(num_cols * ratio) for ratio in N_FEATURES_RATIOS]
    print('n_features_options:', n_features_options)
    feature_reduction_grid = [
        {
            'reduce': [
                #PCA(),
                NMF()
            ],
            'reduce__n_components': n_features_options,
        },
        #{
        #    'reduce': [SelectKBest()],
        #    'reduce__score_func': [
        #        f_regression,
        #        mutual_info_regression
        #    ],
        #    'reduce__k': n_features_options,
        #},
    ]


    # Random forest features
    # Number of trees
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # TODO: search over more params
    regression_grid = [
        #{
        #    'regress': [
        #        #KNeighborsRegressor(),
        #        #linear_model.ARDRegression(),
        #        #linear_model.BayesianRidge(),
        #        #linear_model.ElasticNet(),
        #        #linear_model.LassoLars(),
        #        #linear_model.LinearRegression(),
        #        #linear_model.Ridge(),
        #        #linear_model.SGDRegressor(),
        #        #tree.DecisionTreeRegressor(),
        #        #ensemble.AdaBoostRegressor(),
        #        #ensemble.BaggingRegressor(),
        #        #ensemble.GradientBoostingRegressor(),
        #    ]
        #},
        #{
        #    'regress': [ensemble.RandomForestRegressor()],
        #    #'regress__n_estimators': n_estimators,
        #    #'regress__max_features': max_features,
        #    #'regress__max_depth': max_depth,
        #    #'regress__min_samples_split': min_samples_split,
        #    #'regress__min_samples_leaf': min_samples_leaf,
        #    #'regress__bootstrap': bootstrap
        #},
        {
            'regress': [PLSRegression()]
        }
	#{
        #    'regress': [svm.NuSVR()],
        #    'regress__C': [1, 10, 100, 1000],
        #    'regress__kernel': ['rbf', 'linear', 'poly'],
        #},
	#{
        #    'regress': [svm.LinearSVR()],
        #    'regress__C': [1, 10, 100, 1000]
        #}
        #{
        #    'regress': [neural_network.MLPRegressor()],
        #    'regress__hidden_layer_sizes': [(100,)]
        #},
    ]

    pipeline = Pipeline(
        [
            #('fft', FFTEncoder()),
            #('reduce', DummyEstimator()),
            ('regress', DummyEstimator())
        ],
        #memory=memory
    )

    grid_steps = [
        #feature_reduction_grid,
        regression_grid
    ]
    combined_grids = get_combined_grids(grid_steps)
    print('combined_grids:')
    pprint(combined_grids)

    kfold = KFold(n_splits=NUM_FOLDS or num_rows, random_state=0)
    search = GridSearchCV(
        pipeline,
        combined_grids,
        error_score=np.nan,
        verbose=5,
        n_jobs=-1,
        cv=kfold
    )

    print('*' * 40)
    print('Searching')
    print('*' * 40)
    start = timer()
    search.fit(X, Y)
    end = timer()
    print('Finished in: {}'.format(end - start))

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    best_index = search.best_index_
    best_std = search.cv_results_['std_test_score'][best_index]
    print('best_estimator:', best_estimator)
    print('best_params:', best_params)
    print('best_score:', best_score)
    print('best_std:', best_std)

    return Pipeline([('features', features), ('estimator', best_estimator)])

get_best_estimator__cached = memory.cache(get_best_estimator)


if __name__ == '__main__':
    main()

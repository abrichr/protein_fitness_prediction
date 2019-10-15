# Goals

> Please use the provided subset of this sequence->fitness dataset
> (representing 0.3% of the possible sequence diversity) to:
>
> - construct 10 new sequences outside of the provided dataset with the highest
> predicted fitness for a subsequent round of synthesis.
>
> - Suggest 10 compounds outside of the dataset provided that if you had
> experimental data on would most improve your model (and thus answer
> for #1). 

# Methods 

This implementation resembles that of
[innov’SAR](https://www.genome.jp/aaindex/).

Steps:

## 1) Encoding

### a) [AAIndex](https://www.genome.jp/aaindex/)1 features

Via [PyBioMed](https://github.com/gadsbyfly/PyBioMed):

> This database holds more than 500 numerical indices representing various
> physicochemical and biochemical properties for the 20 standard amino
> acids and correlations between these indices are also listed.

TODO: use correlations

### b) Fast Fourier Transform (FFT)

Via `numpy.fft`.

## 2) Modelling

> the modelling phase, will use the experimental values of the target
> activity, in conjunction with these protein spectra, in order to identify
> a predictive model.
> The model is constructed by the application of standard regression
> approaches based on a learning step and a validation step. innov’SAR
> used a partial least square regression, PLS, as algorithm of regression
> to do the model for the predictions of the enantioselectivity of epoxide
> hydrolase. 

This implementation differs in the following ways:

- Encode the given sequences of length 4 (instead of 9)
- Predict the given fitness values (instead of enantioselectivity of epoxide
hydrolase)
- Use K-Fold cross validation Instead of Leave-One-Out in order to reduce
training time (at the expense of a more biased model).

## 3) Validation

> The root mean squared error (RMSE) and the coefficient of determination
> (R2) are the performance parameters to assess a regression model, during
> the validation step.

The cross validation procedure described in the paper is implemented here
with sklearn.

# Results 

Once the optimal model is selected, it can be used to predict the fitness
of all other possible sequences, and the top 10 with the highest fitness values
can be synthesized for the next round.

10 compounds outside of the provided dataset whose experimental data would most
improve the model are likely those that lie far away from the provided ones
in the mutation graph. These can be determined by constructing an exhaustive
mutation graph of the whole space and sampling from regions that are far away
from each other and from the provided dataset.

# Future Work

- Leave one out cross validation would produce less biased estimators than
K-fold.

- Training on more than 0.3% of the data would likely improve performance.

- Training more models (e.g. SVM, Random Forest) with various hyperparameter
settings might yield better accuracy.

- Creating a mutation graph of the given variants and training separate
classifiers on connected components might have the effect of reducing the
search space for each model, thereby improving performance. This could be
implemented via a 4D boolean adjacency matrix and
`scipy.ndimage.measurements.find_objects`.
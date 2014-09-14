import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/fork4/statsmodels")

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family

def test_linear_model():

    np.random.seed(3425)
    n = 200

    exog = np.random.normal(size=(n, 5))
    e_endog = np.dot(exog, np.r_[0, 1, 0, -2, 0])
    endog = e_endog + np.random.normal(size=n)

    mod1 = OLS(endog, exog)
    params1 = mod1.fit_regularized(alpha=0, return_object=False)

    mod2 = OLS(endog, exog)
    params2 = mod2.fit_regularized(alpha=0.2, return_object=False)

    #e_endog = np.dot(exog, np.r_[0, 1, 0, -2, 0])
    #e_endog = 1 / (1 + np.exp(-e_endog))
    #endog = 1*(np.random.uniform(size=n) < e_endog)

    #bi = family.Binomial()
    #mod1 = GLM(endog, exog, family=bi)
    #params1 = mod1.fit_regularized()
    1/0

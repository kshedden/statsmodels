"""
Linear mixed effects models for Statsmodels

The data are partitioned into disjoint groups.  The probability model
for group i is:

Y = X*beta + Z*gamma + epsilon

where

* n_i is the number of observations in group i
* Y is a n_i dimensional response vector
* X is a n_i x k_fe design matrix for the fixed effects
* beta is a k_fe-dimensional vector of fixed effects slopes
* Z is a n_i x k_re design matrix for the random effects
* gamma is a k_re-dimensional random vector with mean 0
  and covariance matrix Psi; note that each group
  gets its own independent realization of gamma.
* epsilon is a n_i dimensional vector of iid normal
  errors with mean 0 and variance sigma^2; the epsilon
  values are independent both within and between groups

Y, X and Z must be entirely observed.  beta, Psi, and sigma^2 are
estimated using ML or REML estimation, and gamma and epsilon are
random so define the probability model.

The mean structure is E[Y|X,Z] = X*beta.  If only the mean structure
is of interest, GEE is a good alternative to mixed models.

The primary reference for the implementation details is:

MJ Lindstrom, DM Bates (1988).  "Newton Raphson and EM algorithms for
linear mixed effects models for repeated measures data".  Journal of
the American Statistical Association. Volume 83, Issue 404, pages
1014-1022.

See also this more recent document:

http://econ.ucsb.edu/~doug/245a/Papers/Mixed%20Effects%20Implement.pdf

All the likelihood, gradient, and Hessian calculations closely follow
Lindstrom and Bates.

The following two documents are written more from the perspective of
users:

http://lme4.r-forge.r-project.org/lMMwR/lrgprt.pdf

http://lme4.r-forge.r-project.org/slides/2009-07-07-Rennes/3Longitudinal-4.pdf

Notation:

* `cov_re` is the random effects covariance matrix (referred to above
  as Psi) and `scale` is the (scalar) error variance.  For a single
  group, the marginal covariance matrix of endog given exog is scale*I
  + Z * cov_re * Z', where Z is the design matrix for the random
  effects in one group.

Notes:

1. Three different parameterizations are used here in different
places.  The regression slopes (usually called `fe_params`) are
identical in all three parameterizations, but the variance parameters
differ.  The parameterizations are:

* The "natural parameterization" in which cov(endog) = scale*I + Z *
  cov_re * Z', as described above.  This is the main parameterization
  visible to the user.

* The "profile parameterization" in which cov(endog) = I +
  Z * cov_re1 * Z'.  This is the parameterization of the profile
  likelihood that is maximized to produce parameter estimates.
  (see Lindstrom and Bates for details).  The "natural" cov_re is
  equal to the "profile" cov_re1 times scale.

* The "square root parameterization" in which we work with the
  Cholesky factor of cov_re1 instead of cov_re1 directly.

All three parameterizations can be "packed" by concatenating fe_params
together with the lower triangle of the dependence structure.  Note
that when unpacking, it is important to either square or reflect the
dependence structure depending on which parameterization is being
used.

2. The situation where the random effects covariance matrix is
singular is numerically challenging.  Small changes in the covariance
parameters may lead to large changes in the likelihood and
derivatives.

3. The optimization strategy is to first use OLS to get starting
values for the mean structure.  Then we optionally perform a few
steepest ascent steps.  This is followed by conjugate gradient
optimization using one of the scipy gradient optimizers.  The steepest
ascent steps are used to get adequate starting values for the
conjugate gradient optimization, which is much faster.
"""

import numpy as np
import statsmodels.base.model as base
from scipy.optimize import fmin_ncg, fmin_cg, fmin_bfgs, fmin
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools import data as data_tools
from scipy.stats.distributions import norm
import pandas as pd
import patsy
from statsmodels.compat.collections import OrderedDict
from statsmodels.compat import range
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.base._penalties import Penalty
from statsmodels.compat.numpy import np_matrix_rank

from pandas import DataFrame


def _get_exog_re_names(self, exog_re):
    """
    Passes through if given a list of names. Otherwise, gets pandas names
    or creates some generic variable names as needed.
    """
    if self.k_re == 0:
        return []
    if isinstance(exog_re, pd.DataFrame):
        return exog_re.columns.tolist()
    elif isinstance(exog_re, pd.Series) and exog_re.name is not None:
        return [exog_re.name]
    elif isinstance(exog_re, list):
        return exog_re
    return ["Z{0}".format(k + 1) for k in range(exog_re.shape[1])]


class MixedLMParams(object):
    """
    This class represents a parameter state for a mixed linear model.

    Parameters
    ----------
    k_fe : integer
        The number of covariates with fixed effects.
    k_re : integer
        The number of covariates with random coefficients (excluding
        variance components).
    k_vc : integer
        The number of variance components parameters.

    Notes
    -----
    This object represents the parameter state for the model in which
    the scale parameter has been profiled out.
    """

    def __init__(self, k_fe, k_re, k_vc):

        self.k_fe = k_fe
        self.k_re = k_re
        self.k_re2 = k_re * (k_re + 1) / 2
        self.k_vc = k_vc
        self.k_tot = self.k_fe + self.k_re2 + self.k_vc
        self._ix = np.tril_indices(self.k_re)


    def from_packed(params, k_fe, k_re, use_sqrt, with_fe):
        """
        Create a MixedLMParams object from packed parameter vector.

        Parameters
        ----------
        params : array-like
            The mode parameters packed into a single vector.
        k_fe : integer
            The number of covariates with fixed effects
        k_re : integer
            The number of covariates with random effects (excluding
            variance components).
        use_sqrt : boolean
            If True, the random effects covariance matrix is provided
            as its Cholesky factor, otherwise the lower triangle of
            the covariance matrix is stored.
        with_fe : boolean
            If True, `params` contains fixed effects parameters.
            Otherwise, the fixed effects parameters are set to zero.

        Returns
        -------
        A MixedLMParams object.
        """
        k_re2 = int(k_re * (k_re + 1) / 2)

        # The number of covariance parameters.
        if with_fe:
            k_vc = len(params) - k_fe - k_re2
        else:
            k_vc = len(params) - k_re2

        pa = MixedLMParams(k_fe, k_re, k_vc)

        cov_re = np.zeros((k_re, k_re))
        ix = pa._ix
        if with_fe:
            pa.fe_params = params[0:k_fe]
            cov_re[ix] = params[k_fe:k_fe+k_re2]
        else:
            pa.fe_params = np.zeros(k_fe)
            cov_re[ix] = params[0:k_re2]

        if use_sqrt:
            cov_re = np.dot(cov_re, cov_re.T)
        else:
            cov_re = (cov_re + cov_re.T) - np.diag(np.diag(cov_re))

        pa.cov_re = cov_re
        if k_vc > 0:
            if use_sqrt:
                pa.vcomp = params[-k_vc:]**2
            else:
                pa.vcomp = params[-k_vc:]
        else:
            pa.vcomp = np.array([])

        return pa

    from_packed = staticmethod(from_packed)

    def from_components(fe_params=None, cov_re=None, cov_re_sqrt=None, vcomp=None):
        """
        Create a MixedLMParams object from each parameter component.

        Parameters
        ----------
        fe_params : array-like
            The fixed effects parameter (a 1-dimensional array).  If
            None, there are no fixed effects.
        cov_re : array-like
            The random effects covariance matrix (a square, symmetric
            2-dimensional array).
        cov_re_sqrt : array-like
            The Cholesky (lower triangular) square root of the random
            effects covariance matrix.
        vcomp : array-like
            The variance component parameters.  If None, there are no
            variance components.

        Returns
        -------
        A MixedLMParams object.
        """

        if vcomp is None:
            vcomp = np.empty(0)
        if fe_params is None:
            fe_params = np.empty(0)
        if cov_re is None and cov_re_sqrt is None:
            cov_re = np.empty((0, 0))

        k_fe = len(fe_params)
        k_vc = len(vcomp)
        k_re = cov_re.shape[0] if cov_re is not None else cov_re_sqrt.shape[0]

        pa = MixedLMParams(k_fe, k_re, k_vc)
        pa.fe_params = fe_params
        if cov_re_sqrt is not None:
            pa.cov_re = np.dot(cov_re_sqrt, cov_re_sqrt.T)
        elif cov_re is not None:
            pa.cov_re = cov_re

        pa.vcomp = vcomp

        return pa

    from_components = staticmethod(from_components)

    def copy(self):
        """
        Returns a copy of the object.
        """
        obj = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
        obj.fe_params = self.fe_params.copy()
        obj.cov_re = self.cov_re.copy()
        obj.vcomp = self.vcomp.copy()
        return obj


    def get_packed(self, use_sqrt=None, with_fe=False):
        """
        Returns the parameters packed into a single vector.

        Parameters
        ----------
        use_sqrt : None or bool
            If None, `use_sqrt` has the value of this instance's
            `use_sqrt`.  Otherwise it is set to the given value.
        """

        if self.k_re > 0:
            if use_sqrt:
                L = np.linalg.cholesky(self.cov_re)
                cpa = L[self._ix]
            else:
                cpa = self.cov_re[self._ix]
        else:
            cpa = np.zeros(0)

        if use_sqrt:
            vcomp = np.sqrt(self.vcomp)
        else:
            vcomp = self.vcomp

        if with_fe:
            pa = np.concatenate((self.fe_params, cpa, vcomp))
        else:
            pa = np.concatenate((cpa, vcomp))

        return pa


# This is a global switch to use direct linear algebra calculations
# for solving factor-structured linear systems and calculating
# factor-structured determinants.  If False, use the
# Sherman-Morrison-Woodbury update which is more efficient for
# factor-structured matrices.  Should be False except when testing.
_no_smw = False

def _smw_solve(s, A, AtA, B, BI, rhs):
    """
    Solves the system (s*I + A*B*A') * x = rhs for x and returns x.

    Parameters
    ----------
    s : scalar
        See above for usage
    A : square symmetric ndarray
        See above for usage
    AtA : square ndarray
        A.T * A
    B : square symmetric ndarray
        See above for usage
    BI : square symmetric ndarray
        The inverse of `B`.  Can be None if B is singular
    rhs : ndarray
        See above for usage

    Returns
    -------
    x : ndarray
        See above

    If the global variable `_no_smw` is True, this routine uses direct
    linear algebra calculations.  Otherwise it uses the
    Sherman-Morrison-Woodbury identity to speed up the calculation.
    """

    # Direct calculation
    if _no_smw or BI is None:
        mat = np.dot(A, np.dot(B, A.T))
        # Add constant to diagonal
        mat.flat[::mat.shape[0]+1] += s
        return np.linalg.solve(mat, rhs)

    # Use SMW identity
    qmat = BI + AtA / s
    u = np.dot(A.T, rhs)
    qmat = np.linalg.solve(qmat, u)
    qmat = np.dot(A, qmat)
    rslt = rhs / s - qmat / s**2
    return rslt


def _smw_logdet(s, A, AtA, B, BI, B_logdet):
    """
    Use the matrix determinant lemma to accelerate the calculation of
    the log determinant of s*I + A*B*A'.

    Parameters
    ----------
    s : scalar
        See above for usage
    A : square symmetric ndarray
        See above for usage
    AtA : square matrix
        A.T * A
    B : square symmetric ndarray
        See above for usage
    BI : square symmetric ndarray
        The inverse of `B`; can be None if B is singular.
    B_logdet : real
        The log determinant of B

    Returns
    -------
    The log determinant of s*I + A*B*A'.
    """

    p = A.shape[0]

    if _no_smw or BI is None:
        mat = np.dot(A, np.dot(B, A.T))
        # Add constant to diagonal
        mat.flat[::p+1] += s
        _, ld = np.linalg.slogdet(mat)
        return ld

    ld = p * np.log(s)

    qmat = BI + AtA / s
    _, ld1 = np.linalg.slogdet(qmat)

    return B_logdet + ld + ld1


class MixedLM(base.LikelihoodModel):
    """
    An object specifying a linear mixed effects model.  Use the `fit`
    method to fit the model and obtain a results object.

    Parameters
    ----------
    endog : 1d array-like
        The dependent variable
    exog : 2d array-like
        A matrix of covariates used to determine the
        mean structure (the "fixed effects" covariates).
    groups : 1d array-like
        A vector of labels determining the groups -- data from
        different groups are independent
    exog_re : 2d array-like
        A matrix of covariates used to determine the variance and
        covariance structure (the "random effects" covariates).  If
        None, defaults to a random intercept for each group.
    exog_vc : dict-like
        TODO
    use_sqrt : bool
        If True, optimization is carried out using the lower
        triangle of the square root of the random effects
        covariance matrix, otherwise it is carried out using the
        lower triangle of the random effects covariance matrix.
    missing : string
        The approach to missing data handling

    Notes
    -----
    The covariates in `exog`, `exog_re` and `exog_vx` may (but need
    not) partially or wholly overlap.

    `use_sqrt` should almost always be set to True.  The main use case
    for use_sqrt=False is when complicated patterns of fixed values in
    the covariance structure are set (using the `free` argument to
    `fit`) that cannot be expressed in terms of the Cholesky factor L.
    """

    def __init__(self, endog, exog, groups, exog_re=None,
                 exog_vc=None, use_sqrt=True, missing='none',
                 **kwargs):

        self.use_sqrt = use_sqrt

        # Some defaults
        self.reml = True
        self.fe_pen = None
        self.re_pen = None

        # Needs to run early so that the names are sorted.
        self._setup_vcomp(exog_vc)

        # If there is one covariate, it may be passed in as a column
        # vector, convert these to 2d arrays.
        # TODO: Can this be moved up in the class hierarchy?
        #       yes, it should be done up the hierarchy
        if (exog is not None and
                data_tools._is_using_ndarray_type(exog, None) and
                exog.ndim == 1):
            exog = exog[:, None]
        if (exog_re is not None and
                data_tools._is_using_ndarray_type(exog_re, None) and
                exog_re.ndim == 1):
            exog_re = exog_re[:, None]

        # Calling super creates self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(MixedLM, self).__init__(endog, exog, groups=groups,
                                      exog_re=exog_re, missing=missing,
                                      **kwargs)

        self._init_keys.extend(["use_sqrt"])

        self.k_fe = exog.shape[1] # Number of fixed effects parameters

        if exog_re is None and exog_vc is None:
            # Default random effects structure (random intercepts).
            self.k_re = 1
            self.k_re2 = 1
            self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
            self.data.exog_re = self.exog_re
            self.data.param_names = self.exog_names + ['Intercept RE']

        elif exog_re is not None:
            # Process exog_re the same way that exog is handled
            # upstream
            # TODO: this is wrong and should be handled upstream wholly
            self.data.exog_re = exog_re
            self.exog_re = np.asarray(exog_re)
            if self.exog_re.ndim == 1:
                self.exog_re = self.exog_re[:, None]
            # Model dimensions
            # Number of random effect covariates
            self.k_re = self.exog_re.shape[1]
            # Number of covariance parameters
            self.k_re2 = self.k_re * (self.k_re + 1) // 2

        else:
            # All random effects are variance components
            self.k_re = 0
            self.k_re2 = 0

        if not self.data._param_names:
            # HACK: could've been set in from_formula already
            # needs refactor
            (param_names, exog_re_names,
                 exog_re_names_full) = self._make_param_names(exog_re)
            self.data.param_names = param_names
            self.data.exog_re_names = exog_re_names
            self.data.exog_re_names_full = exog_re_names_full

        self.k_params = self.k_fe + self.k_re2

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        for i,g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)

        # Split the data by groups
        self.endog_li = self.group_list(self.endog)
        self.exog_li = self.group_list(self.exog)
        self.exog_re_li = self.group_list(self.exog_re)

        # Precompute this.
        if self.exog_re is None:
            self.exog_re2_li = None
        else:
            self.exog_re2_li = [np.dot(x.T, x) for x in self.exog_re_li]

        # The total number of observations, summed over all groups
        self.n_totobs = sum([len(y) for y in self.endog_li])
        # why do it like the above?
        self.nobs = len(self.endog)

        # Set the fixed effects parameter names
        if self.exog_names is None:
            self.exog_names = ["FE%d" % (k + 1) for k in
                               range(self.exog.shape[1])]


    def _setup_vcomp(self, exog_vc):
        if exog_vc is None:
            exog_vc = {}
        self.exog_vc = exog_vc
        self.k_vc = len(exog_vc)
        vc_names = list(set(exog_vc.keys()))
        vc_names.sort()
        self._vc_names = vc_names


    def _make_param_names(self, exog_re):
        """
        Returns the full parameter names list, just the exogenous random
        effects variables, and the exogenous random effects variables with
        the interaction terms.
        """
        exog_names = list(self.exog_names)
        exog_re_names = _get_exog_re_names(self, exog_re)
        param_names = []

        jj = self.k_fe
        for i in range(len(exog_re_names)):
            for j in range(i + 1):
                if i == j:
                    param_names.append(exog_re_names[i] + " RE")
                else:
                    param_names.append(exog_re_names[j] + " RE x " +
                                       exog_re_names[i] + " RE")
                jj += 1

        vc_names = self._vc_names

        return exog_names + param_names + vc_names, exog_re_names, param_names

    @classmethod
    def from_formula(cls, formula, data, re_formula=None, subset=None,
                     *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model
        data : array-like
            The data for the model. See Notes.
        re_formula : string
            A one-sided formula defining the variance structure of the
            model.  The default gives a random intercept for each
            group.
        subset : array-like
            An array-like object of booleans, integers, or index
            values that indicate the subset of df to use in the
            model. Assumes df is a `pandas.DataFrame`
        args : extra arguments
            These are passed to the model
        kwargs : extra keyword arguments
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model : Model instance

        Notes
        ------
        `data` must define __getitem__ with the keys in the formula
        terms args and kwargs are passed on to the model
        instantiation. E.g., a numpy structured or rec array, a
        dictionary, or a pandas DataFrame.

        If `re_formula` is not provided, the default is a random
        intercept for each group.

        This method currently does not correctly handle missing
        values, so missing values should be explicitly dropped from
        the DataFrame before calling this method.
        """

        if "groups" not in kwargs.keys():
            raise AttributeError("'groups' is a required keyword argument in MixedLM.from_formula")

        # If `groups` is a variable name, retrieve the data for the
        # groups variable.
        if type(kwargs["groups"]) == str:
            kwargs["groups"] = np.asarray(data[kwargs["groups"]])

        if re_formula is not None:
            eval_env = kwargs.get('eval_env', None)
            if eval_env is None:
                eval_env = 1
            elif eval_env == -1:
                from patsy import EvalEnvironment
                eval_env = EvalEnvironment({})
            exog_re = patsy.dmatrix(re_formula, data, eval_env=eval_env)
            exog_re_names = exog_re.design_info.column_names
            exog_re = np.asarray(exog_re)
        else:
            exog_re = np.ones((data.shape[0], 1),
                              dtype=np.float64)
            exog_re_names = ["Intercept"]

        mod = super(MixedLM, cls).from_formula(formula, data,
                                               subset=None,
                                               exog_re=exog_re,
                                               *args, **kwargs)

        # expand re names to account for pairs of RE
        (param_names,
         exog_re_names,
         exog_re_names_full) = mod._make_param_names(exog_re_names)

        mod.data.param_names = param_names
        mod.data.exog_re_names = exog_re_names
        mod.data.exog_re_names_full = exog_re_names_full
        mod.data.vcomp_names = mod._vc_names

        return mod


    def group_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        grouping structure.
        """

        if array is None:
            return None

        if array.ndim == 1:
            return [np.array(array[self.row_indices[k]])
                    for k in self.group_labels]
        else:
            return [np.array(array[self.row_indices[k], :])
                    for k in self.group_labels]


    def fit_regularized(self, start_params=None, method='l1', alpha=0,
                        ceps=1e-4, ptol=1e-6, maxit=200, **fit_kwargs):
        """
        Fit a model in which the fixed effects parameters are
        penalized.  The dependence parameters are held fixed at their
        estimated values in the unpenalized model.

        Parameters
        ----------
        method : string of Penalty object
            Method for regularization.  If a string, must be 'l1'.
        alpha : array-like
            Scalar or vector of penalty weights.  If a scalar, the
            same weight is applied to all coefficients; if a vector,
            it contains a weight for each coefficient.  If method is a
            Penalty object, the weights are scaled by alpha.  For L1
            regularization, the weights are used directly.
        ceps : positive real scalar
            Fixed effects parameters smaller than this value
            in magnitude are treaded as being zero.
        ptol : positive real scalar
            Convergence occurs when the sup norm difference
            between successive values of `fe_params` is less than
            `ptol`.
        maxit : integer
            The maximum number of iterations.
        fit_kwargs : keywords
            Additional keyword arguments passed to fit.

        Returns
        -------
        A MixedLMResults instance containing the results.

        Notes
        -----
        The covariance structure is not updated as the fixed effects
        parameters are varied.

        The algorithm used here for L1 regularization is a"shooting"
        or cyclic coordinate descent algorithm.

        If method is 'l1', then `fe_pen` and `cov_pen` are used to
        obtain the covariance structure, but are ignored during the
        L1-penalized fitting.

        References
        ----------
        Friedman, J. H., Hastie, T. and Tibshirani, R. Regularized
        Paths for Generalized Linear Models via Coordinate
        Descent. Journal of Statistical Software, 33(1) (2008)
        http://www.jstatsoft.org/v33/i01/paper

        http://statweb.stanford.edu/~tibs/stat315a/Supplements/fuse.pdf
        """

        if type(method) == str and (method.lower() != 'l1'):
            raise ValueError("Invalid regularization method")

        # If method is a smooth penalty just optimize directly.
        if isinstance(method, Penalty):
            # Scale the penalty weights by alpha
            method.alpha = alpha
            fit_kwargs.update({"fe_pen": method})
            return self.fit(**fit_kwargs)

        if np.isscalar(alpha):
            alpha = alpha * np.ones(self.k_fe, dtype=np.float64)

        # Fit the unpenalized model to get the dependence structure.
        mdf = self.fit(**fit_kwargs)
        fe_params = mdf.fe_params
        cov_re = mdf.cov_re
        scale = mdf.scale
        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        for itr in range(maxit):

            fe_params_s = fe_params.copy()
            for j in range(self.k_fe):

                if abs(fe_params[j]) < ceps:
                    continue

                # The residuals
                fe_params[j] = 0.
                expval = np.dot(self.exog, fe_params)
                resid_all = self.endog - expval

                # The loss function has the form
                # a*x^2 + b*x + pwt*|x|
                a, b = 0., 0.
                for k, lab in enumerate(self.group_labels):

                    exog = self.exog_li[k]
                    ex_r = self.exog_re_li[k]
                    ex2_r = self.exog_re2_li[k]
                    resid = resid_all[self.row_indices[lab]]

                    x = exog[:,j]
                    u = _smw_solve(scale, ex_r, ex2_r, cov_re,
                                   cov_re_inv, x)
                    a += np.dot(u, x)
                    b -= 2 * np.dot(u, resid)

                pwt1 = alpha[j]
                if b > pwt1:
                    fe_params[j] = -(b - pwt1) / (2 * a)
                elif b < -pwt1:
                    fe_params[j] = -(b + pwt1) / (2 * a)

            if np.abs(fe_params_s - fe_params).max() < ptol:
                break

        # Replace the fixed effects estimates with their penalized
        # values, leave the dependence parameters in their unpenalized
        # state.
        params_prof = mdf.params.copy()
        params_prof[0:self.k_fe] = fe_params

        scale = self.get_scale(fe_params, mdf.cov_re_unscaled, mdf.vcomp)

        # Get the Hessian including only the nonzero fixed effects,
        # then blow back up to the full size after inverting.
        hess = self.hessian_full(params_prof)
        pcov = np.nan * np.ones_like(hess)
        ii = np.abs(params_prof) > ceps
        ii[self.k_fe:] = True
        ii = np.flatnonzero(ii)
        hess1 = hess[ii, :][:, ii]
        pcov[np.ix_(ii,ii)] = np.linalg.inv(-hess1)

        params_object = MixedLMParams.from_components(fe_params, cov_re=cov_re)

        results = MixedLMResults(self, params_prof, pcov / scale)
        results.params_object = params_object
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.scale = scale
        results.cov_re_unscaled = mdf.cov_re_unscaled
        results.method = mdf.method
        results.converged = True
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc

        return MixedLMResultsWrapper(results)


    def get_fe_params(self, cov_re, vcomp):
        """
        Use GLS to update the fixed effects parameter estimates.

        Parameters
        ----------
        cov_re : array-like
            The covariance matrix of the random effects.

        Returns
        -------
        The GLS estimates of the fixed effects parameters.
        """

        if self.k_fe == 0:
            return np.array([])

        if self.k_re == 0:
            cov_re_inv = np.zeros((0,0))
        else:
            cov_re_inv = np.linalg.inv(cov_re)

        # Cache these quantities that don't change.
        if not hasattr(self, "_endex_li"):
            self._endex_li = [None] * self.n_groups

        xtxy = 0.
        for i in range(self.n_groups):

            cov_aug, cov_aug_inv, _ = self._augment_cov_re(cov_re, cov_re_inv, vcomp, i)

            if self._endex_li[i] is None:
                mat = np.concatenate((self.exog_li[i], self.endog_li[i][:, None]), axis=1)
                self._endex_li[i] = mat

            exog = self.exog_li[i]
            ex_r, ex2_r = self._augment_exog(i)
            u = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, self._endex_li[i])
            xtxy += np.dot(exog.T, u)

        fe_params = np.linalg.solve(xtxy[:, 0:-1], xtxy[:, -1])

        return fe_params


    def _reparam(self):
        """
        Returns parameters of the map converting parameters from the
        form used in optimization to the form returned to the user.

        Returns
        -------
        lin : list-like
            Linear terms of the map
        quad : list-like
            Quadratic terms of the map

        Notes
        -----
        If P are the standard form parameters and R are the
        transformed parameters (i.e. with the Cholesky square root
        covariance and square root transformed variane components),
        then P[i] = lin[i] * R + R' * quad[i] * R
        """

        k_fe, k_re, k_re2, k_vc = self.k_fe, self.k_re, self.k_re2, self.k_vc
        k_tot = k_fe + k_re2 + k_vc
        ix = np.tril_indices(self.k_re)

        lin = []
        for k in range(k_fe):
            e = np.zeros(k_tot)
            e[k] = 1
            lin.append(e)
        for k in range(k_re2):
            lin.append(np.zeros(k_tot))
        for k in range(k_vc):
            lin.append(np.zeros(k_tot))

        quad = []
        # Quadratic terms for fixed effects.
        for k in range(k_tot):
            quad.append(np.zeros((k_tot, k_tot)))

        # Quadratic terms for random effects covariance.
        ii = np.tril_indices(k_re)
        ix = [(a,b) for a,b in zip(ii[0], ii[1])]
        for i1 in range(k_re2):
            for i2 in range(k_re2):
                ix1 = ix[i1]
                ix2 = ix[i2]
                if (ix1[1] == ix2[1]) and (ix1[0] <= ix2[0]):
                    ii = (ix2[0], ix1[0])
                    k = ix.index(ii)
                    quad[k_fe+k][k_fe+i2, k_fe+i1] += 1
        for k in range(k_tot):
            quad[k] = 0.5*(quad[k] + quad[k].T)

        # Quadratic terms for variance components.
        km = k_fe + k_re2
        for k in range(km, km+k_vc):
            quad[k][k, k] = 1

        return lin, quad


    # This isn't useful for anything.  For standard error calculation
    # we only need hessian_full.  For optimization this is not the
    # right Hessian because it does account for the profiling over
    # fe_params.  The correct Hessian for optimization is too
    # difficult and expensive to calculate.
    def hessian_sqrt(self, params):
        """
        Returns the Hessian matrix of the log-likelihood evaluated at
        a given point, calculated with respect to the parameterization
        in which the random effects covariance matrix is represented
        through its Cholesky square root.

        Parameters
        ----------
        params : MixedLMParams or array-like
            The model parameters.  If array-like, must contain packed
            parameters that are compatible with this model.

        Returns
        -------
        The Hessian matrix of the profile log likelihood function,
        evaluated at `params`.

        Notes
        -----
        If `params` is provided as a MixedLMParams object it may be of
        any parameterization.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re)

        score_fe0, score_re0, score_vc0 = self.score_full(params)
        score0 = np.concatenate((score_fe0, score_re0, score_vc0))
        hess0 = self.hessian_full(params)

        params_vec = params.get_packed(use_sqrt=True, with_fe=True)

        lin, quad = self._reparam()
        k_tot = self.k_fe + self.k_re2

        # Convert Hessian to new coordinates
        hess = 0.
        for i in range(k_tot):
            hess += 2 * score0[i] * quad[i]
        for i in range(k_tot):
            vi = lin[i] + 2*np.dot(quad[i], params_vec)
            for j in range(k_tot):
                vj = lin[j] + 2*np.dot(quad[j], params_vec)
                hess += hess0[i, j] * np.outer(vi, vj)

        return hess


    def _augment_cov_re(self, cov_re, cov_re_inv, vcomp, group):
        """
        Returns the covariance matrix and its inverse for all random
        effects.  Also returns the adjustment to the determinant for
        this group.
        """
        if self.k_vc == 0:
            return cov_re, cov_re_inv, 0.

        vc_var = []
        for j,k in enumerate(self._vc_names):
            if group not in self.exog_vc[k]:
                continue
            vc_var.append([vcomp[j]] * self.exog_vc[k][group].shape[1])
        vc_var = np.concatenate(vc_var)

        m = cov_re.shape[0] + len(vc_var)
        cov_aug = np.zeros((m, m))
        cov_aug[0:self.k_re, 0:self.k_re] = cov_re
        ix = np.arange(self.k_re, m)
        cov_aug[ix, ix] = vc_var

        cov_aug_inv = np.zeros((m, m))
        cov_aug_inv[0:self.k_re, 0:self.k_re] = cov_re_inv
        cov_aug_inv[ix, ix] = 1 / vc_var

        return cov_aug, cov_aug_inv, np.sum(np.log(vc_var))


    def _augment_exog(self, group_ix):
        """
        Concatenate the columns for variance components to the columns
        for other random effects to obtain a single random effects
        exog matrix.  Returns the matrix and its cross product matrix.
        """
        ex_r = self.exog_re_li[group_ix] if self.k_re > 0 else None
        ex2_r = self.exog_re2_li[group_ix] if self.k_re > 0 else None
        if self.k_vc == 0:
            return ex_r, ex2_r

        group = self.group_labels[group_ix]
        ex = [ex_r] if self.k_re > 0 else []
        for j,k in enumerate(self._vc_names):
            if group not in self.exog_vc[k]:
                continue
            ex.append(self.exog_vc[k][group])
        ex = np.concatenate(ex, axis=1)

        ex2 = np.dot(ex.T, ex)
        return ex, ex2


    def loglike(self, params, profile_fe=True):
        """
        Evaluate the (profile) log-likelihood of the linear mixed
        effects model.

        Parameters
        ----------
        params : MixedLMParams, or array-like.
            The parameter value.  If array-like, must be a packed
            parameter vector containing only the covariance
            parameters.
        profile_fe : boolean
            If True, replace the provided value of `params_fe` with
            the GLS estimates.

        Returns
        -------
        The log-likelihood value at `params`.

        Notes
        -----
        This is the profile likelihood in which the scale parameter
        `scale` has been profiled out.

        The input parameter state, if provided as a MixedLMParams
        object, can be with respect to any parameterization.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe,
                                               self.k_re, self.use_sqrt,
                                               with_fe=False)

        cov_re = params.cov_re
        vcomp = params.vcomp

        # Move to the profile set
        if profile_fe:
            fe_params = self.get_fe_params(cov_re, vcomp)
        else:
            fe_params = params.fe_params

        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = None
            _, cov_re_logdet = np.linalg.slogdet(cov_re)
        else:
            cov_re_inv = np.zeros((0, 0))
            cov_re_logdet = 0

        # The residuals
        expval = np.dot(self.exog, fe_params)
        resid_all = self.endog - expval

        likeval = 0.

        # Handle the covariance penalty
        if self.cov_pen is not None:
            likeval -= self.cov_pen.func(cov_re, cov_re_inv)

        # Handle the fixed effects penalty
        if self.fe_pen is not None:
            likeval -= self.fe_pen.func(fe_params)

        xvx, qf = 0., 0.
        for k, group in enumerate(self.group_labels):

            cov_aug, cov_aug_inv, adj_logdet = self._augment_cov_re(cov_re, cov_re_inv, vcomp, group)
            cov_aug_logdet = cov_re_logdet + adj_logdet

            exog = self.exog_li[k]
            ex_r, ex2_r = self._augment_exog(k)
            resid = resid_all[self.row_indices[group]]

            # Part 1 of the log likelihood (for both ML and REML)
            ld = _smw_logdet(1., ex_r, ex2_r, cov_aug, cov_aug_inv, cov_aug_logdet)
            likeval -= ld / 2.

            # Part 2 of the log likelihood (for both ML and REML)
            u = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, resid)
            qf += np.dot(resid, u)

            # Adjustment for REML
            if self.reml:
                mat = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv,
                                 exog)
                xvx += np.dot(exog.T, mat)

        if self.reml:
            likeval -= (self.n_totobs - self.k_fe) * np.log(qf) / 2.
            _,ld = np.linalg.slogdet(xvx)
            likeval -= ld / 2.
            likeval -= (self.n_totobs - self.k_fe) * np.log(2 * np.pi) / 2.
            likeval += ((self.n_totobs - self.k_fe) *
                        np.log(self.n_totobs - self.k_fe) / 2.)
            likeval -= (self.n_totobs - self.k_fe) / 2.
        else:
            likeval -= self.n_totobs * np.log(qf) / 2.
            likeval -= self.n_totobs * np.log(2 * np.pi) / 2.
            likeval += self.n_totobs * np.log(self.n_totobs) / 2.
            likeval -= self.n_totobs / 2.

        return likeval


    def _gen_dV_dPsi(self, ex_r, group, max_ix=None):
        """
        A generator that yields the derivative of the covariance
        matrix V (=I + Z*Psi*Z') with respect to the free elements of
        Psi.  Each call to the generator yields the index of Psi with
        respect to which the derivative is taken, and the derivative
        matrix with respect to that element of Psi.  Psi is a
        symmetric matrix, so the free elements are the lower triangle.
        If max_ix is not None, the iterations terminate after max_ix
        values are yielded.
        """

        # Regular random effects
        jj = 0
        for j1 in range(self.k_re):
            for j2 in range(j1 + 1):
                if max_ix is not None and jj > max_ix:
                    return
                mat_l, mat_r = ex_r[:,j1:j1+1], ex_r[:,j2:j2+1] # Need 2d
                yield jj, mat_l, mat_r, j1 == j2
                jj += 1

        # Variance components
        for ky in self._vc_names:
            if max_ix is not None and jj > max_ix:
                return
            if group in self.exog_vc[ky]:
                mat = self.exog_vc[ky][group]
                yield jj, mat, mat, True
                jj += 1


    def score(self, params, profile_fe=True):
        """
        Returns the score vector of the profile log-likelihood.

        Notes
        -----
        The score vector that is returned is computed with respect to
        the parameterization defined by this model instance's
        `use_sqrt` attribute.  The input value `params` can be with
        respect to any parameterization.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe,
                                               self.k_re, self.use_sqrt,
                                               with_fe=False)

        if profile_fe:
            params.fe_params = self.get_fe_params(params.cov_re, params.vcomp)

        if self.use_sqrt:
            score_fe, score_re, score_vc = self.score_sqrt(params)
        else:
            score_fe, score_re, score_vc = self.score_full(params)

        if self._freepat is not None:
            score_fe *= self._freepat.fe_params
            score_re *= self._freepat.cov_re[self._freepat._ix]
            # free not used for variance components

        if profile_fe:
            return np.concatenate((score_re, score_vc))
        else:
            return np.concatenate((score_fe, score_re, score_vc))


    def hessian(self, params):
        """
        Returns the Hessian matrix of the profile log-likelihood.

        Notes
        -----
        The Hessian matrix that is returned is computed with respect
        to the parameterization defined by this model's `use_sqrt`
        attribute.  The input value `params` can be with respect to
        any parameterization.
        """

        if self.use_sqrt:
            hess = self.hessian_sqrt(params)
        else:
            hess = self.hessian_full(params)

        return hess


    def score_full(self, params):
        """
        Calculates the score vector for the profiled log-likelihood of
        the mixed effects model with respect to the parameterization
        in which the random effects covariance matrix is represented
        in its full form (not using the Cholesky factor).

        Parameters
        ----------
        params : MixedLMParams or array-like
            The parameter at which the score function is evaluated.
            If array-like, must contain the packed covariance matrix,
            without fe_params.

        Returns
        -------
        The score vector, calculated at `params`.

        Notes
        -----
        The score vector that is returned is taken with respect to the
        parameterization in which `cov_re` is represented through its
        lower triangle (without taking the Cholesky square root).

        The input, if provided as a MixedLMParams object, can be of
        any parameterization.
        """

        fe_params = params.fe_params
        cov_re = params.cov_re
        vcomp = params.vcomp

        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        score_fe = np.zeros(self.k_fe)
        score_re = np.zeros(self.k_re2)
        score_vc = np.zeros(self.k_vc)

        # Handle the covariance penalty.
        if self.cov_pen is not None:
            score_re -= self.cov_pen.grad(cov_re, cov_re_inv)

        # Handle the fixed effects penalty.
        if self.fe_pen is not None:
            score_fe -= self.fe_pen.grad(fe_params)

        # resid' V^{-1} resid, summed over the groups (a scalar)
        rvir = 0.

        # exog' V^{-1} resid, summed over the groups (a k_fe
        # dimensional vector)
        xtvir = 0.

        # exog' V^{_1} exog, summed over the groups (a k_fe x k_fe
        # matrix)
        xtvix = 0.

        # V^{-1} exog' dV/dQ_jj exog V^{-1}, where Q_jj is the jj^th
        # covariance parameter.
        xtax = [0.,] * (self.k_re2 + self.k_vc)

        # Temporary related to the gradient of log |V|
        dlv = np.zeros(self.k_re2 + self.k_vc)

        # resid' V^{-1} dV/dQ_jj V^{-1} resid (a scalar)
        rvavr = np.zeros(self.k_re2 + self.k_vc)

        for k, lab in enumerate(self.group_labels):

            cov_aug, cov_aug_inv, _ = self._augment_cov_re(cov_re, cov_re_inv, vcomp, lab)

            exog = self.exog_li[k]
            ex_r, ex2_r = self._augment_exog(k)

            # The residuals
            resid = self.endog_li[k]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            if self.reml:
                viexog = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, exog)
                xtvix += np.dot(exog.T, viexog)

            # Contributions to the covariance parameter gradient
            jj = 0
            vex = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, ex_r)
            vir = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, resid)
            for jj, matl, matr, sym in self._gen_dV_dPsi(ex_r, lab):
                vsl = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, matl)
                dlv[jj] = np.sum(matr * vsl) # trace dot
                if not sym:
                    vsr = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, matr)
                    dlv[jj] += np.sum(matl * vsr) # trace dot

                ul = np.dot(vir, matl)
                ur = ul.T if sym else np.dot(matr.T, vir)
                ulr = np.dot(ul, ur)
                rvavr[jj] += ulr
                if not sym:
                    rvavr[jj] += ulr.T

                if self.reml:
                    ul = np.dot(viexog.T, matl)
                    ur = ul.T if sym else np.dot(matr.T, viexog)
                    ulr = np.dot(ul, ur)
                    xtax[jj] += ulr
                    if not sym:
                        xtax[jj] += ulr.T

            # Contribution of log|V| to the covariance parameter
            # gradient.
            if self.k_re > 0:
                score_re -= 0.5 * dlv[0:self.k_re2]
            if self.k_vc > 0:
                score_vc -= 0.5 * dlv[self.k_re2:]

            # Needed for the fixed effects params gradient
            rvir += np.dot(resid, vir)
            xtvir += np.dot(exog.T, vir)

        fac = self.n_totobs
        if self.reml:
            fac -= self.k_fe

        score_fe += fac * xtvir / rvir

        if self.k_re > 0:
            score_re += 0.5 * fac * rvavr[0:self.k_re2] / rvir
        if self.k_vc > 0:
            score_vc += 0.5 * fac * rvavr[self.k_re2:] / rvir

        if self.reml:
            for j in range(self.k_re2):
                score_re[j] += 0.5 * np.trace(np.linalg.solve(xtvix, xtax[j]))
            for j in range(self.k_vc):
                score_vc[j] += 0.5 * np.trace(np.linalg.solve(xtvix, xtax[self.k_re2 + j]))

        return score_fe, score_re, score_vc


    def score_sqrt(self, params):
        """
        Returns the score vector with respect to the parameterization
        in which the random effects covariance matrix is represented
        through its Cholesky square root.

        Parameters
        ----------
        params : MixedLMParams or array-like
            The model parameters.  If array-like must contain packed
            parameters that are compatible with this model instance.

        Returns
        -------
        The score vector.

        Notes
        -----
        The input, if provided as a MixedLMParams object, can be of
        any parameterization.
        """

        score_fe, score_re, score_vc = self.score_full(params)
        params_vec = params.get_packed(use_sqrt=True, with_fe=True)

        lin, quad = self._reparam()

        score_full = np.concatenate((score_fe, score_re, score_vc))
        scr = 0.
        for i in range(len(params_vec)):
            v = lin[i] + 2 * np.dot(quad[i], params_vec)
            scr += score_full[i] * v
        score_fe = scr[0:self.k_fe]
        score_re = scr[self.k_fe:self.k_fe + self.k_re2]
        score_vc = scr[self.k_fe + self.k_re2:]

        return score_fe, score_re, score_vc


    def hessian_full(self, params):
        """
        Calculates the Hessian matrix for the mixed effects model with
        respect to the parameterization in which the covariance matrix
        is represented directly (without square-root transformation).

        Parameters
        ----------
        params : MixedLMParams or array-like
            The model parameters at which the Hessian is calculated.
            If array-like, must contain the packed parameters in a
            form that is compatible with this model instance.

        Returns
        -------
        hess : 2d ndarray
            The Hessian matrix, evaluated at `params`.
        """

        if type(params) is not MixedLMParams:
            params = MixedLMParams.from_packed(params, self.k_fe, self.k_re,
                                               use_sqrt=self.use_sqrt,
                                               with_fe=True)

        fe_params = params.fe_params
        vcomp = params.vcomp
        cov_re = params.cov_re
        if self.k_re > 0:
            try:
                cov_re_inv = np.linalg.inv(cov_re)
            except np.linalg.LinAlgError:
                cov_re_inv = None
        else:
            cov_re_inv = np.empty(0)

        # Blocks for the fixed and random effects parameters.
        hess_fe = 0.
        hess_re = np.zeros((self.k_re2 + self.k_vc, self.k_re2 + self.k_vc))
        hess_fere = np.zeros((self.k_re2 + self.k_vc, self.k_fe))

        fac = self.n_totobs
        if self.reml:
            fac -= self.exog.shape[1]

        rvir = 0.
        xtvix = 0.
        xtax = [0.,] * (self.k_re2 + self.k_vc)
        m = self.k_re2 + self.k_vc
        B = np.zeros(m)
        D = np.zeros((m, m))
        F = [[0.] * m for k in range(m)]
        for k, group in enumerate(self.group_labels):

            cov_aug, cov_aug_inv, _ = self._augment_cov_re(cov_re, cov_re_inv, vcomp, group)

            exog = self.exog_li[k]
            ex_r, ex2_r = self._augment_exog(k)

            # The residuals
            resid = self.endog_li[k]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            viexog = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, exog)
            xtvix += np.dot(exog.T, viexog)
            vir = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, resid)
            rvir += np.dot(resid, vir)

            for jj1,matl1,matr1,sym1 in self._gen_dV_dPsi(ex_r, group):

                ul = np.dot(viexog.T, matl1)
                ur = np.dot(matr1.T, vir)
                hess_fere[jj1, :] += np.dot(ul, ur)
                if not sym1:
                    ul = np.dot(viexog.T, matr1)
                    ur = np.dot(matl1.T, vir)
                    hess_fere[jj1, :] += np.dot(ul, ur)

                if self.reml:
                    ul = np.dot(viexog.T, matl1)
                    ur = ul if sym1 else np.dot(viexog.T, matr1)
                    ulr = np.dot(ul, ur.T)
                    xtax[jj1] += ulr
                    if not sym1:
                        xtax[jj1] += ulr.T

                ul = np.dot(vir, matl1)
                ur = ul if sym1 else np.dot(vir, matr1)
                B[jj1] += np.dot(ul, ur) * (1 if sym1 else 2)

                # V^{-1} * dV/d_theta
                E = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, matl1)
                E = np.dot(E, matr1.T)
                if not sym1:
                    E1 = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, matr1)
                    E += np.dot(E1, matl1.T)

                for jj2,matl2,matr2,sym2 in self._gen_dV_dPsi(ex_r, group, jj1):

                    re = np.dot(matr2.T, E)
                    rev = np.dot(re, vir[:, None])
                    vl = np.dot(vir[None, :], matl2)
                    vt = 2*np.dot(vl, rev)

                    if not sym2:
                        le = np.dot(matl2.T, E)
                        lev = np.dot(le, vir[:, None])
                        vr = np.dot(vir[None, :], matr2)
                        vt += 2*np.dot(vr, lev)

                    D[jj1, jj2] += vt
                    if jj1 != jj2:
                        D[jj2, jj1] += vt

                    R = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, matl2)
                    rt = np.sum(R * re.T) / 2 # trace dot
                    if not sym2:
                        R = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, matr2)
                        rt += np.sum(R * le.T) / 2 # trace dot

                    hess_re[jj1, jj2] += rt
                    if jj1 != jj2:
                        hess_re[jj2, jj1] += rt

                    if self.reml:
                        ev = np.dot(E, viexog)
                        u1 = np.dot(viexog.T, matl2)
                        u2 = np.dot(matr2.T, ev)
                        um = np.dot(u1, u2)
                        F[jj1][jj2] += um + um.T
                        if not sym2:
                            u1 = np.dot(viexog.T, matr2)
                            u2 = np.dot(matl2.T, ev)
                            um = np.dot(u1, u2)
                            F[jj1][jj2] += um + um.T

        hess_fe -= fac * xtvix / rvir
        hess_re = hess_re - 0.5 * fac * (D/rvir - np.outer(B, B) / rvir**2)
        hess_fere = -fac * hess_fere / rvir

        if self.reml:
            QL = [np.linalg.solve(xtvix, x) for x in xtax]
            for j1 in range(self.k_re2 + self.k_vc):
                for j2 in range(j1 + 1):
                    a = np.sum(QL[j1].T * QL[j2]) # trace dot
                    a -= np.trace(np.linalg.solve(xtvix, F[j1][j2]))
                    a *= 0.5
                    hess_re[j1, j2] += a
                    if j1 > j2:
                        hess_re[j2, j1] += a

        # Put the blocks together to get the Hessian.
        m = self.k_fe + self.k_re2 + self.k_vc
        hess = np.zeros((m, m))
        hess[0:self.k_fe, 0:self.k_fe] = hess_fe
        hess[0:self.k_fe, self.k_fe:] = hess_fere.T
        hess[self.k_fe:, 0:self.k_fe] = hess_fere
        hess[self.k_fe:, self.k_fe:] = hess_re

        return hess


    def steepest_ascent(self, params, n_iter):
        """
        Take steepest ascent steps to increase the log-likelihood
        function.

        Parameters
        ----------
        params : array-like
            The starting point of the optimization.
        n_iter: non-negative integer
            Number of iterations to perform.

        Returns
        -------
        A MixedLMParameters object containing the final value of the
        optimization.
        """

        if n_iter == 0:
            return params

        fval = self.loglike(params)

        cov_re = params.cov_re
        if self.k_re > 0:
            if self.use_sqrt:
                cov_re_sqrt = np.linalg.cholesky(cov_re)
                pa = cov_re_sqrt[params._ix]
            else:
                pa = cov_re[params._ix]
        else:
            cov_re_sqrt = np.zeros((0, 0))
            pa = np.zeros(0)

        for itr in range(n_iter):

            grad = self.score(pa)
            grad = grad / np.max(np.abs(grad))

            sl = 0.5
            while sl > 1e-20:
                pa1 = pa + sl*grad
                fval1 = self.loglike(pa1)
                if fval1 > fval:
                    pa = pa1
                    fval = fval1
                sl /= 2

        return pa


    def get_scale(self, fe_params, cov_re, vcomp):
        """
        Returns the estimated error variance based on given estimates
        of the slopes and random effects covariance matrix.

        Parameters
        ----------
        fe_params : array-like
            The regression slope estimates
        cov_re : 2d array-like
            Estimate of the random effects covariance matrix
        vcomp : array-like
            Estimate of the variance components

        Returns
        -------
        scale : float
            The estimated error variance.
        """

        try:
            cov_re_inv = np.linalg.inv(cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        qf = 0.
        for k in range(self.n_groups):

            cov_aug, cov_aug_inv, _ = self._augment_cov_re(cov_re, cov_re_inv, vcomp, k)

            exog = self.exog_li[k]
            ex_r, ex2_r = self._augment_exog(k)

            # The residuals
            resid = self.endog_li[k]
            if self.k_fe > 0:
                expval = np.dot(exog, fe_params)
                resid = resid - expval

            mat = _smw_solve(1., ex_r, ex2_r, cov_aug, cov_aug_inv, resid)
            qf += np.dot(resid, mat)

        if self.reml:
            qf /= (self.n_totobs - self.k_fe)
        else:
            qf /= self.n_totobs

        return qf


    def fit(self, start_params=None, reml=True, niter_sa=0,
            do_cg=True, fe_pen=None, cov_pen=None, free=None,
            full_output=False, **kwargs):
        """
        Fit a linear mixed model to the data.

        Parameters
        ----------
        start_params: array-like or MixedLMParams
            If a `MixedLMParams` the state provides the starting
            value.  If array-like, this is the packed parameter
            vector, assumed to be in the same state as this model.
        reml : bool
            If true, fit according to the REML likelihood, else
            fit the standard likelihood using ML.
        niter_sa : integer
            The number of steepest ascent iterations
        do_cg : bool
            If True, a conjugate gradient algorithm is
            used for optimization (following any steepest
            descent steps).
        cov_pen : CovariancePenalty object
            A penalty for the random effects covariance matrix
        fe_pen : Penalty object
            A penalty on the fixed effects
        free : MixedLMParams object
            If not `None`, this is a mask that allows parameters to be
            held fixed at specified values.  A 1 indicates that the
            correspondinig parameter is estimated, a 0 indicates that
            it is fixed at its starting value.  Setting the `cov_re`
            component to the identity matrix fits a model with
            independent random effects.  The state of `use_sqrt` for
            `free` must agree with that of the parent model.
        full_output : bool
            If true, attach iteration history to results

        Returns
        -------
        A MixedLMResults instance.

        Notes
        -----
        If `start` is provided as an array, it must have the same
        `use_sqrt` state as the parent model.

        The value of `free` must have the same `use_sqrt` state as the
        parent model.
        """

        self.reml = reml
        self.cov_pen = cov_pen
        self.fe_pen = fe_pen

        self._freepat = free

        if full_output:
            hist = []
        else:
            hist = None

        success = False

        params = MixedLMParams(self.k_fe, self.k_re, self.k_vc)
        params.fe_params = np.zeros(self.k_fe)
        params.cov_re = np.eye(self.k_re)
        params.vcomp = np.ones(self.k_vc)

        # Try up to 10 times to make the optimization work.  Usually
        # only one cycle is used.
        if do_cg:
            for cycle in range(10):

                params = self.steepest_ascent(params, niter_sa)
                try:
                    kwargs["retall"] = hist is not None
                    if "disp" not in kwargs:
                        kwargs["disp"] = False
                    # Only bfgs and lbfgs seem to work
                    kwargs["method"] = "bfgs"
                    packed = params.get_packed(use_sqrt=self.use_sqrt, with_fe=False)
                    rslt = super(MixedLM, self).fit(start_params=packed,
                                                    skip_hessian=True,
                                                    **kwargs)
                except np.linalg.LinAlgError:
                    continue

                # The optimization succeeded
                params = rslt.params
                if hist is not None:
                    hist.append(rslt.mle_retvals)
                break

        converged = rslt.mle_retvals['converged']
        if not converged:
            msg = "Gradient optimization failed."
            warnings.warn(msg, ConvergenceWarning)

        # Convert to the final parameterization (i.e. undo the square
        # root transform of the covariance matrix, and the profiling
        # over the error variance).
        params = MixedLMParams.from_packed(params, self.k_fe, self.k_re,
                                           use_sqrt=self.use_sqrt, with_fe=False)
        cov_re_unscaled = params.cov_re
        vcomp_unscaled = params.vcomp
        fe_params = self.get_fe_params(cov_re_unscaled, vcomp_unscaled)
        params.fe_params = fe_params
        scale = self.get_scale(fe_params, cov_re_unscaled, vcomp_unscaled)
        cov_re = scale * cov_re_unscaled
        vcomp = scale * vcomp_unscaled

        if (((self.k_re > 0) and (np.min(np.abs(np.diag(cov_re))) < 0.01)) or
            ((self.k_vc > 0) and (np.min(np.abs(vcomp)) < 0.01))):
            msg = "The MLE may be on the boundary of the parameter space."
            warnings.warn(msg, ConvergenceWarning)

        # Compute the Hessian at the MLE.  Note that this is the
        # Hessian with respect to the random effects covariance matrix
        # (not its square root).  It is used for obtaining standard
        # errors, not for optimization.
        hess = self.hessian_full(params)
        if free is not None:
            pcov = np.zeros_like(hess)
            pat = self._freepat.get_packed(with_fe=True)
            ii = np.flatnonzero(pat)
            if len(ii) > 0:
                hess1 = hess[np.ix_(ii, ii)]
                pcov[np.ix_(ii, ii)] = np.linalg.inv(-hess1)
        else:
            pcov = np.linalg.inv(-hess)

        # Prepare a results class instance
        params_packed = params.get_packed(use_sqrt=False, with_fe=True)
        results = MixedLMResults(self, params_packed, pcov / scale)
        results.params_object = params
        results.fe_params = fe_params
        results.cov_re = cov_re
        results.vcomp = vcomp
        results.scale = scale
        results.cov_re_unscaled = cov_re_unscaled
        results.method = "REML" if self.reml else "ML"
        results.converged = converged
        results.hist = hist
        results.reml = self.reml
        results.cov_pen = self.cov_pen
        results.k_fe = self.k_fe
        results.k_re = self.k_re
        results.k_re2 = self.k_re2
        results.k_vc = self.k_vc
        results.use_sqrt = self.use_sqrt
        results.freepat = self._freepat

        return MixedLMResultsWrapper(results)


class MixedLMResults(base.LikelihoodModelResults):
    '''
    Class to contain results of fitting a linear mixed effects model.

    MixedLMResults inherits from statsmodels.LikelihoodModelResults

    Parameters
    ----------
    See statsmodels.LikelihoodModelResults

    Returns
    -------
    **Attributes**

    model : class instance
        Pointer to PHreg model instance that called fit.
    normalized_cov_params : array
        The sampling covariance matrix of the estimates
    fe_params : array
        The fitted fixed-effects coefficients
    re_params : array
        The fitted random-effects covariance matrix
    bse_fe : array
        The standard errors of the fitted fixed effects coefficients
    bse_re : array
        The standard errors of the fitted random effects covariance
        matrix

    See Also
    --------
    statsmodels.LikelihoodModelResults
    '''

    def __init__(self, model, params, cov_params):

        super(MixedLMResults, self).__init__(model, params,
                                             normalized_cov_params=cov_params)
        self.nobs = self.model.nobs
        self.df_resid = self.nobs - np_matrix_rank(self.model.exog)

    @cache_readonly
    def bse_fe(self):
        """
        Returns the standard errors of the fixed effect regression
        coefficients.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(np.diag(self.cov_params())[0:p])

    @cache_readonly
    def bse_re(self):
        """
        Returns the standard errors of the variance parameters.  Note
        that the sampling distribution of variance parameters is
        strongly skewed unless the sample size is large, so these
        standard errors may not give meaningful confidence intervals
        of p-values if used in the usual way.
        """
        p = self.model.exog.shape[1]
        return np.sqrt(self.scale * np.diag(self.cov_params())[p:])

    @cache_readonly
    def random_effects(self):
        """
        Returns the conditional means of all random effects given the
        data.

        Returns
        -------
        random_effects : DataFrame
            A DataFrame with the distinct `group` values as the index
            and the conditional means of the random effects
            in the columns.
        """
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        ranef_dict = {}
        for k in range(self.model.n_groups):

            endog = self.model.endog_li[k]
            exog = self.model.exog_li[k]
            ex_r = self.model.exog_re_li[k]
            ex2_r = self.model.exog_re2_li[k]
            label = self.model.group_labels[k]

            # Get the residuals
            expval = np.dot(exog, self.fe_params)
            resid = endog - expval

            vresid = _smw_solve(self.scale, ex_r, ex2_r, self.cov_re,
                                cov_re_inv, resid)

            ranef_dict[label] = np.dot(self.cov_re,
                                       np.dot(ex_r.T, vresid))

        column_names = dict(zip(range(self.k_re),
                                      self.model.data.exog_re_names))
        df = DataFrame.from_dict(ranef_dict, orient='index')
        return df.rename(columns=column_names).ix[self.model.group_labels]

    @cache_readonly
    def random_effects_cov(self):
        """
        Returns the conditional covariance matrix of the random
        effects for each group given the data.

        Returns
        -------
        random_effects_cov : dict
            A dictionary mapping the distinct values of the `group`
            variable to the conditional covariance matrix of the
            random effects given the data.
        """

        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            cov_re_inv = None

        ranef_dict = {}
        #columns = self.model.data.exog_re_names
        for k in range(self.model.n_groups):

            ex_r = self.model.exog_re_li[k]
            ex2_r = self.model.exog_re2_li[k]
            label = self.model.group_labels[k]

            mat1 = np.dot(ex_r, self.cov_re)
            mat2 = _smw_solve(self.scale, ex_r, ex2_r, self.cov_re,
                              cov_re_inv, mat1)
            mat2 = np.dot(mat1.T, mat2)

            ranef_dict[label] = self.cov_re - mat2
            #ranef_dict[label] = DataFrame(self.cov_re - mat2,
            #                              index=columns, columns=columns)


        return ranef_dict

    def summary(self, yname=None, xname_fe=None, xname_re=None,
                title=None, alpha=.05):
        """
        Summarize the mixed model regression results.

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname_fe : list of strings, optional
            Fixed effects covariate names
        xname_re : list of strings, optional
            Random effects covariate names
        title : string, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results
        """

        from statsmodels.iolib import summary2
        smry = summary2.Summary()

        info = OrderedDict()
        info["Model:"] = "MixedLM"
        if yname is None:
            yname = self.model.endog_names
        info["No. Observations:"] = str(self.model.n_totobs)
        info["No. Groups:"] = str(self.model.n_groups)

        gs = np.array([len(x) for x in self.model.endog_li])
        info["Min. group size:"] = "%.0f" % min(gs)
        info["Max. group size:"] = "%.0f" % max(gs)
        info["Mean group size:"] = "%.1f" % np.mean(gs)

        info["Dependent Variable:"] = yname
        info["Method:"] = self.method
        info["Scale:"] = self.scale
        info["Likelihood:"] = self.llf
        info["Converged:"] = "Yes" if self.converged else "No"
        smry.add_dict(info)
        smry.add_title("Mixed Linear Model Regression Results")

        float_fmt = "%.3f"

        sdf = np.nan * np.ones((self.k_fe + self.k_re2 + self.k_vc, 6))

        # Coefficient estimates
        sdf[0:self.k_fe, 0] = self.fe_params

        # Standard errors
        sdf[0:self.k_fe, 1] = np.sqrt(np.diag(self.cov_params()[0:self.k_fe]))

        # Z-scores
        sdf[0:self.k_fe, 2] = sdf[0:self.k_fe, 0] / sdf[0:self.k_fe, 1]

        # p-values
        sdf[0:self.k_fe, 3] = 2 * norm.cdf(-np.abs(sdf[0:self.k_fe, 2]))

        # Confidence intervals
        qm = -norm.ppf(alpha / 2)
        sdf[0:self.k_fe, 4] = sdf[0:self.k_fe, 0] - qm * sdf[0:self.k_fe, 1]
        sdf[0:self.k_fe, 5] = sdf[0:self.k_fe, 0] + qm * sdf[0:self.k_fe, 1]

        # All random effects variances and covariances
        jj = self.k_fe
        for i in range(self.k_re):
            for j in range(i + 1):
                sdf[jj, 0] = self.cov_re[i, j]
                sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
                jj += 1

        # Variance components
        for i in range(self.k_vc):
            sdf[jj, 0] = self.vcomp[i]
            sdf[jj, 1] = np.sqrt(self.scale) * self.bse[jj]
            jj += 1

        sdf = pd.DataFrame(index=self.model.data.param_names, data=sdf)
        sdf.columns = ['Coef.', 'Std.Err.', 'z', 'P>|z|',
                          '[' + str(alpha/2), str(1-alpha/2) + ']']
        for col in sdf.columns:
            sdf[col] = [float_fmt % x if np.isfinite(x) else ""
                        for x in sdf[col]]

        smry.add_df(sdf, align='r')

        return smry


    @cache_readonly
    def llf(self):
        return self.model.loglike(self.params_object, profile_fe=False)


    def profile_re(self, re_ix, num_low=5, dist_low=1., num_high=5,
                   dist_high=1.):
        """
        Calculate a series of values along a 1-dimensional profile
        likelihood.

        Parameters
        ----------
        re_ix : integer
            The index of the variance parameter for which to construct
            a profile likelihood.
        num_low : integer
            The number of points at which to calculate the likelihood
            below the MLE of the parameter of interest.
        dist_low : float
            The distance below the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.
        num_high : integer
            The number of points at which to calculate the likelihood
            abov the MLE of the parameter of interest.
        dist_high : float
            The distance above the MLE of the parameter of interest to
            begin calculating points on the profile likelihood.

        Returns
        -------
        An array with two columns.  The first column contains the
        values to which the parameter of interest is constrained.  The
        second column contains the corresponding likelihood values.

        Notes
        -----
        Only variance parameters can be profiled.  `re_ix` is the index
        of the random effect that is profiled.
        """

        pmodel = self.model
        k_fe = pmodel.exog.shape[1]
        k_re = pmodel.exog_re.shape[1]
        k_vc = pmodel.k_vc
        endog, exog, groups = pmodel.endog, pmodel.exog, pmodel.groups

        # Need to permute the columns of the random effects design
        # matrix so that the profiled variable is in the first column.
        ix = np.arange(k_re)
        ix[0] = re_ix
        ix[re_ix] = 0
        exog_re = pmodel.exog_re.copy()[:, ix]

        # Permute the covariance structure to match the permuted
        # design matrix.
        params = self.params_object.copy()
        cov_re_unscaled = params.cov_re
        cov_re_unscaled = cov_re_unscaled[np.ix_(ix, ix)]
        params.cov_re = cov_re_unscaled

        # Convert dist_low and dist_high to the profile
        # parameterization
        cov_re = self.scale * cov_re_unscaled
        low = (cov_re[0, 0] - dist_low) / self.scale
        high = (cov_re[0, 0] + dist_high) / self.scale

        # Define the sequence of values to which the parameter of
        # interest will be constrained.
        ru0 = cov_re_unscaled[0, 0]
        if low <= 0:
            raise ValueError("dist_low is too large and would result in a "
                             "negative variance. Try a smaller value.")
        left = np.linspace(low, ru0, num_low + 1)
        right = np.linspace(ru0, high, num_high+1)[1:]
        rvalues = np.concatenate((left, right))

        # Indicators of which parameters are free and fixed.
        free = MixedLMParams(k_fe, k_re, k_vc)
        if self.freepat is None:
            free.fe_params = np.ones(k_fe)
            free.vcomp = np.ones(k_vc)
            mat = np.ones((k_re, k_re))
        else:
            free.fe_params = self.freepat.fe_params
            free.vcomp = self.freepat.vcomp
            mat = self.freepat.cov_re
            mat = mat[np.ix_(ix, ix)]
        mat[0, 0] = 0
        free.cov_re = mat

        klass = self.model.__class__
        init_kwargs = pmodel._get_init_kwds()
        init_kwargs['exog_re'] = exog_re

        likev = []
        for x in rvalues:

            model = klass(endog, exog, **init_kwargs)

            cov_re = params.cov_re.copy()
            cov_re[0, 0] = x

            # Shrink the covariance parameters until a PSD covariance
            # matrix is obtained.
            dg = np.diag(cov_re).copy()
            success = False
            for ks in range(50):
                try:
                    np.linalg.cholesky(cov_re)
                    success = True
                    break
                except np.linalg.LinAlgError:
                    cov_re /= 2
                    np.fill_diagonal(cov_re, dg)
            if not success:
                raise ValueError("unable to find PSD covariance matrix along likelihood profile")

            params.cov_re = cov_re
            # TODO should use fit_kwargs
            rslt = model.fit(start_params=params, free=free,
                             reml=self.reml, cov_pen=self.cov_pen)._results
            likev.append([rslt.cov_re[0, 0], rslt.llf])

        likev = np.asarray(likev)

        return likev


class MixedLMResultsWrapper(base.LikelihoodResultsWrapper):
    _attrs = {'bse_re': ('generic_columns', 'exog_re_names_full'),
              'fe_params': ('generic_columns', 'xnames'),
              'bse_fe': ('generic_columns', 'xnames'),
              'cov_re': ('generic_columns_2d', 'exog_re_names'),
              'cov_re_unscaled': ('generic_columns_2d', 'exog_re_names'),
              }
    _upstream_attrs = base.LikelihoodResultsWrapper._wrap_attrs
    _wrap_attrs = base.wrap.union_dicts(_attrs, _upstream_attrs)

    _methods = {}
    _upstream_methods = base.LikelihoodResultsWrapper._wrap_methods
    _wrap_methods = base.wrap.union_dicts(_methods, _upstream_methods)

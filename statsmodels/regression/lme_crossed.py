"""
Linear mixed effects with group + non-group covariance structure

scale*(I + exog_re * cov_re * exog_re' + exog_re_x * cov_re_x * exog_re_x')
= scale*V

-n*log(scale)/2 - log(|V|)/2 - 0.5 * R' V^{-1} R / scale

scale = R' V^{-1} R / n

n*log(n)/2 - n*log(R' V^{-1} R) / 2 - n/2
"""

import numpy as np
import statsmodels.base.model as base



class MixedLMX(base.LikelihoodModel):


    def __init__(self, endog, exog, groups, exog_re=None,
                 exog_re_x=None, missing='none'):

        # Some defaults
        self.reml = True

        # If there is one covariate, it may be passed in as a column
        # vector, convert these to 2d arrays.
        # TODO: Can this be moved up in the class hierarchy?
        if exog is not None and exog.ndim == 1:
            exog = exog[:,None]
        if exog_re is not None and exog_re.ndim == 1:
            exog_re = exog_re[:,None]
        if exog_re_x is not None and exog_re_x.ndim == 1:
            exog_re_x = exog_re_x[:,None]

        # Calling super creates self.endog, etc. as ndarrays and the
        # original exog, endog, etc. are self.data.endog, etc.
        super(MixedLM, self).__init__(endog, exog, groups=groups,
                                      exog_re=exog_re,
                                      exog_re_x=exog_re_x,
                                      missing=missing)

        # Number of fixed effects parameters
        self.k_fe = exog.shape[1]

        # Number of group random effect parameters
        if exog_re is not None:

            # Number of random effect covariates
            self.k_re = exog_re.shape[1]

            # Number of group-effect covariance parameters
            self.k_re2 = self.k_re * (self.k_re + 1) // 2

        else:
            self.k_re = 0
            self.k_re2 = 0

        # Number of non-group random effect parameters
        if exog_re_x is not None:

            # Number of random effect covariates
            self.k_re_x = exog_re_x.shape[1]

            # Number of group-effect covariance parameters
            self.k_re_x2 = self.k_re_x * (self.k_re_x + 1) // 2

        else:
            self.k_re_x = 0
            self.k_re_x2 = 0

        # Override the default value
        self.nparams = self.k_fe + self.k_re2 + self.k_re_x2

        # Convert the data to the internal representation, which is a
        # list of arrays, corresponding to the groups.
        # TODO: what if there are no groups?
        group_labels = list(set(groups))
        group_labels.sort()
        row_indices = dict((s, []) for s in group_labels)
        for i,g in enumerate(groups):
            row_indices[g].append(i)
        self.row_indices = row_indices
        self.group_labels = group_labels
        self.n_groups = len(self.group_labels)

        # The total number of observations, summed over all groups
        self.n_totobs = len(self.endog)

        # Set the fixed effects parameter names
        if self.exog_names is None:
            self.exog_names = ["FE%d" % (k + 1) for k in
                               range(self.exog.shape[1])]

        # Set the grouped random effect parameter names
        if isinstance(self.exog_re, pd.DataFrame):
            self.exog_re_names = list(self.exog_re.columns)
        else:
            self.exog_re_names = ["Z%d" % (k+1) for k in
                                  range(self.exog_re.shape[1])]

        # TODO: Set the non-grouped random effect parameter names

        # Indices used to pack/unpack the parameter vector.
        self._cov_re_pos = (self.k_fe, self.k_fe + self.k_re2)
        self._cov_re_x_pos = (self.k_fe + self.k_re2, self.nparams)
        self._cov_re_ix = np.tril_indices(self.k_re)
        self._cov_re_x_ix = np.tril_indices(self.k_re_x)

    def pack(self, fe_params, cov_re_l, cov_re_x_l, full_cov=False):

        if full_cov:
            cov_re_l = np.linalg.cholesky(cov_re_l)
            cov_re_x_l = np.linalg.cholesky(cov_re_x_l)

        params = np.zeros(self.nparams, dtype=np.float64)
        params[0:self.k_fe] = fe_params
        params[self._cov_re_pos[0]:self._cov_re_pos[1]] =\
                       cov_re_l[self._cov_re_ix]
        params[self._cov_re_x_pos[0]:self._cov_re_x_pos[1]] =\
                       cov_re_x_l[self._cov_re_x_ix]

        return params

    def unpack(self, params, return_cov=False):

        cov_re = np.zeros((self.k_re, self.k_re), dtype=np.float64)
        cov_re_x = np.zeros((self.k_re_x, self.k_re_x),
                            dtype=np.float64)
        params_fe = params[0:self.k_fe]
        cov_re[self._cov_re_ix] =\
                  params[self._cov_re_pos[0]:self._cov_re_pos[1]]
        cov_re_x[self._cov_re_x_ix] =\
                  params[self._cov_re_x_pos[0]:self._cov_re_x_pos[1]]
        params[self._cov_re_pos[0]:self._cov_re_pos[1]] = cov_re_l[self._cov_re_ix]
        params[self._cov_re_pos[0]:self._cov_re_pos[1]] = cov_re_l[self._cov_re_ix]

        if return_cov:
            cov_re = np.dot(cov_re, cov_re.T)
            cov_re_x = np.dot(cov_re_x, cov_re_x.T)

        return params, cov_re, cov_re_x

    def _whiten_groups(self, params, rhs):
        """
        Whiten one or more vectors based on the group covariance
        structure.

        Arguments
        ---------
        params : array-like
            The packed parameter vector
        rhs : list
            A list of right-hand-sides to be whitened in-place.
        """

        params, cov_re, cov_re_x = self.unpack(params, return_cov=True)

        for i, ii in self.row_indices:
            exr = self.exog_re[ii, :]
            vmat = np.dot(exr, np.dot(cov_re, exr.T))
            for x in rhs:
                rhs[ii] = np.linalg.solve(vmat, rhs[ii])


    def _whiten(self, params, rhs):
        """
        Whiten in-place each element of a list.

        Arguments
        ---------
        params : array-like
            The packed parameter vector
        rhs : list
            A list of arrays to whiten in-place
        """

        params, cov_re, cov_re_x = self.unpack(params, return_cov=True)

        exog_re_x_w = self.exog_re_x.copy()
        self._whiten_groups(params, [exog_re_x_w,])
        exog_re_x_w = exog_re_x_w[0]

        mat = np.linalg.inv(self.cov_re_x)
        mat += np.dot(self.exog_re_x.T, exog_re_x_w)

        self._whiten_groups(params, rhs)
        rhs1 = [x.copy() for x in rhs]
        rhs2 = [np.dot(self.exog_re_x.T, x) for x in rhs1]
        rhs3 = [np.linalg.solve(mat, x) for x in rhs2]
        rhs4 = [np.dot(exog_re_x_w, x) for x in rhs3]

        for x,y in zip(rhs, rhs4):
            x -= y

    def loglike(self, params):



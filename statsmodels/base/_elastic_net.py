

#        if hasattr(self, "regularized") and self.regularized == True:
#            smry.add_text("Standard errors do not account for the regularization")



def _fit(model, method="coord_descent", maxiter=100,
         alpha=0., L1_wt=1., start_params=None,
         cnvrg_tol=1e-7, zero_tol=1e-8,
         return_object=False, **kwargs):
    """
    Return a regularized fit to a regression model.

    Parameters
    ----------
    method :
        Only the coordinate descent algorithm is implemented.
    maxiter : integer
        The maximum number of iteration cycles (an iteration cycle
        involves running coordinate descent on all variables).
    alpha : scalar or array-like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    start_params : array-like
        Starting values for `params`.
    cnvrg_tol : scalar
        If `params` changes by less than this amount (in sup-norm)
        in once iteration cycle, the algorithm terminates with
        convergence.
    zero_tol : scalar
        Any estimated coefficient smaller than this value is
        replaced with zero.
    return_object : bool
        If False, only the parameter estimates are returned.

    Returns
    -------
    A results object of the same type returned by `fit`.

    Notes
    -----
    The penalty is the "elastic net" penalty, which
    is a convex combination of L1 and L2 penalties.

    The function that is minimized is:

    -loglike/n + alpha*((1-L1_wt)*|params|_2^2/2 + L1_wt*|params|_1)

    where |*|_1 and |*|_2 are the L1 and L2 norms.

    The computational approach used here is to obtain a quadratic
    approximation to the smooth part of the target function:

    -loglike/n + alpha*(1-L1_wt)*|params|_2^2/2

    then optimize the L1 penalized version of this function along
    a coordinate axis.

    This is a generic implementation that may be reimplemented in
    specific models for better performance.
    """

    k_exog = self.exog.shape[1]
    n_exog = self.exog.shape[0]

    if np.isscalar(alpha):
        alpha = alpha * np.ones(k_exog, dtype=np.float64)

    # Define starting params
    if start_params is None:
        params = np.zeros(k_exog)
    else:
        params = start_params.copy()

    # All the negative penalized log-likelihood functions.
    def gen_npfuncs(k):

        def nploglike(params):
            pen = alpha[k]*((1 - L1_wt)*params**2/2 + L1_wt*np.abs(params))
            return -model_1var.loglike(np.r_[params]) / n_exog + pen

        def npscore(params):
            pen_grad = alpha[k]*(1 - L1_wt)*params
            return -model_1var.score(np.r_[params])[0] / n_exog + pen_grad

        def nphess(params):
            pen_hess = alpha[k]*(1 - L1_wt)
            return -model_1var.hessian(np.r_[params])[0,0] / n_exog + pen_hess

        return nploglike, npscore, nphess

    nploglike_funcs = [gen_npfuncs(k) for k in range(len(params))]

    converged = False
    btol = 1e-8
    params_zero = np.zeros(len(params), dtype=bool)

    init_args = {k : getattr(self, k) for k in self._init_keys
                 if k != "offset" and hasattr(self, k)}

    for itr in range(maxiter):

        # Sweep through the parameters
        params_save = params.copy()
        for k in range(k_exog):

            # Under the active set method, if a parameter becomes
            # zero we don't try to change it again.
            if params_zero[k]:
                continue

            # Set the offset to account for the variables that are
            # being held fixed in the current coordinate
            # optimization.
            params0 = params.copy()
            params0[k] = 0
            offset = np.dot(self.exog, params0)
            if hasattr(self, "offset") and self.offset is not None:
                offset += self.offset

            # Create a one-variable model for optimization.
            model_1var = model.__class__(self.endog, self.exog[:, k], offset=offset,
                                         **init_args)

            func, grad, hess = tuple(nploglike_funcs[k])
            params[k] = _opt_1d(func, grad, hess, params[k], alpha[k]*L1_wt,
                                tol=btol)

            # Update the active set
            if itr > 0 and np.abs(params[k]) < zero_tol:
                params_zero[k] = True
                params[k] = 0.

        # Check for convergence
        pchange = np.max(np.abs(params - params_save))
        if pchange < cnvrg_tol:
            converged = True
            break

    # Set approximate zero coefficients to be exactly zero
    params *= np.abs(params) >= zero_tol

    if not return_object:
        return params

    # Fit the reduced model to get standard errors and other
    # post-estimation results.
    ii = np.flatnonzero(params)
    cov = np.zeros((k_exog, k_exog), dtype=np.float64)
    if len(ii) > 0:
        model = self.__class__(self.endog, self.exog[:, ii],
                               **kwargs)
        rslt = model.fit()
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params
    else:
        model = self.__class__(self.endog, self.exog[:, 0],
                               **kwargs)
        rslt = model.fit()
        cov[np.ix_(ii, ii)] = rslt.normalized_cov_params

    # fit may return a results or a results wrapper
    if issubclass(rslt.__class__, wrap.ResultsWrapper):
        klass = rslt._results.__class__
    else:
        klass = rslt.__class__
    rfit = klass(self, params, cov)
    rfit.regularized = True


def _opt_1d(func, grad, hess, start, L1_wt, tol):
    """
    Optimize a L1-penalized smooth one-dimensional function of a
    single variable.

    Parameters:
    -----------
    func : function
        A smooth function of a single variable to be optimized
        with L1 penaty.
    grad : function
        The gradient of `func`.
    hess : function
        The Hessian of `func`.
    start : real
        A starting value for the function argument
    L1_wt : non-negative real
        The weight for the L1 penalty function.
    tol : non-negative real
        A convergence threshold.

    Returns:
    --------
    The argmin of the objective function.
    """

    # TODO: can we detect failures without calling func twice?

    from scipy.optimize import brent

    x = start
    f = func(x)
    b = grad(x)
    c = hess(x)
    d = b - c*x

    if L1_wt > np.abs(d):
        return 0.
    elif d >= 0:
        x += (L1_wt - b) / c
    elif d < 0:
        x -= (L1_wt + b) / c

    f1 = func(x)

    # This is an expensive fall-back if the quadratic
    # approximation is poor and sends us far off-course.
    if f1 > f + 1e-10:
        return brent(func, brack=(x-0.2, x+0.2), tol=tol)

    return x

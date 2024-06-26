import numpy as np
import scipy.stats as stats
from typing import Tuple, Callable

class stats_models:
    @staticmethod
    def logLikelihood_NormalCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Normal model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        mu = vec[0]
        sigma = vec[1]
        if sigma > 0:
            dif = stats.norm.cdf(DATMAT[:, 1], loc=mu, scale=sigma) - stats.norm.cdf(DATMAT[:, 0], loc=mu, scale=sigma)
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def logLikelihood_LogNormalCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Log-Normal model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        mu = vec[0]
        sigma = vec[1]
        if sigma > 0:
            dif = stats.lognorm.cdf(DATMAT[:, 1], s=sigma, scale=np.exp(mu)) - stats.lognorm.cdf(DATMAT[:, 0], s=sigma, scale=np.exp(mu))
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def logLikelihood_GammaCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Gamma model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        alpha = vec[0]
        mu = vec[1]
        beta = mu / alpha
        if alpha > 0 and mu > 0:
            dif = stats.gamma.cdf(DATMAT[:, 1], shape=alpha, scale=beta) - stats.gamma.cdf(DATMAT[:, 0], shape=alpha, scale=beta)
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def logLikelihood_InverseGaussianCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Inverse Gaussian model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        mu = vec[0]
        lambda_ = vec[1]
        if mu > 0 and lambda_ > 0:
            dif = stats.invgauss.cdf(DATMAT[:, 1], mu=mu, scale=lambda_) - stats.invgauss.cdf(DATMAT[:, 0], mu=mu, scale=lambda_)
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def logLikelihood_ExponentialCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Exponential model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        theta = vec
        if theta > 0:
            dif = stats.expon.cdf(DATMAT[:, 1], rate=1/theta) - stats.expon.cdf(DATMAT[:, 0], rate=1/theta)
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def logLikelihood_WeibullCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Weibull model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        beta = vec[0]
        sigma = vec[1]
        if beta > 0 and sigma > 0:
            dif = stats.weibull_min.cdf(DATMAT[:, 1], shape = beta, scale=sigma) - stats.weibull_min.cdf(DATMAT[:, 0], shape = beta, scale=sigma)
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def logLikelihood_GumbelCEN(vec: np.array, DATMAT: np.array) -> float:
        """
        Compute the log-likelihood of the Censored Gumbel model.
        :param vec: the parameter vector of the model.
        :rtype vec: np.array.
        :param DATMAT: the data matrix.
        :rtype DATMAT: np.array.
        :return: the log-likelihood of the model.
        """
        eps = 1e-6
        a = vec[0]
        b = vec[1]
        if b > 0:
            dif = stats.gumbel_r.cdf(DATMAT[:, 1], shape=a, scale=b) - stats.gumbel_r.cdf(DATMAT[:, 0], shape=a, scale=b)
            dif[dif < eps] = eps
            lv = np.sum(np.log(dif))
        else:
            lv = -np.inf
        return lv
    
    @staticmethod
    def Weibetmle2(BETA: float, XX: np.array, NN: float, TT: float) -> float:
        """
        Compute the log-likelihood of the Censored Weibull model.
        :param BETA: the shape parameter of the Weibull distribution.
        :rtype BETA: float.
        :param XX: the data matrix.
        :rtype XX: np.array.
        :param NN: the number of observations.
        :rtype NN: float.
        :param TT: the censoring threshold.
        :rtype TT: float.
        :return: the log-likelihood of the model.
        """
        if BETA > 0:
            k1 = np.sum(np.exp(BETA * XX))
            k2 = np.sum(XX * np.exp(BETA * XX))
            sali = (NN / BETA) + TT - (NN * k2 / k1)
        else:
            sali = 5555
        return sali
import numpy as np
from typing import Tuple, Callable, Literal

from scipy.stats import beta,invgauss, expon, gamma, norm, weibull_min, gumbel_r
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class graphs:
    @staticmethod
    def empirical_cdf(data:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        """Compute the empirical CDF
        :param data: data
        :return: x, F(x)
        """
        n = len(data)
        X = np.sort(data)
        F = np.arange(1,n+1)/(n+1)
        return X, F
    
    @classmethod
    def PP_plot(cls,data:np.ndarray,
                params:np.ndarray,
                dist:Literal['Normal',
                             'Lognormal',
                             'Exponential',
                             'Gamma',
                             'Weibull',
                             'Inverse Gaussian'
                             'Gumbel'
                             ]='Normal')->None:
        
        """Create a PP plot for the Model
        :param data: data
        :rtype: np.ndarray
        :return: None
        """
        n = len(data)
        
        # Compute trust bands
        enes = np.arange(1, n+1)
        tau=1-(.05/n)
        tau1=(1-tau)/2
        tau2=(1+tau)/2
        aenes=n+1-enes
        Ban1 = beta.ppf(tau1, enes, aenes)
        Ban2 = beta.ppf(tau2, enes, aenes)
        
        # Compute the empirical CDF
        y_ecdf, sorted_data = cls.empirical_cdf(data)
        # Compute the theoretical CDF
        if dist == 'Normal':
            mu, lambd = params
            ww = invgauss.cdf(sorted_data,mu = mu/lambd, scale = lambd,loc = 0)
        elif dist == 'Lognormal':
            mu, sigma = params
            ww = norm.cdf(np.log(sorted_data),loc=mu,scale=sigma)
        elif dist == 'Exponential':
            theta = params[0]
            ww = expon.cdf(sorted_data,scale=theta)
        elif dist == 'Gamma':
            alpha, beta = params
            ww = gamma.cdf(sorted_data,alpha,scale=1/beta)
        elif dist == 'Weibull':
            c, loc, scale = params
            ww = weibull_min.cdf(sorted_data,c,loc,scale)
        elif dist == 'Inverse Gaussian':
            mu, lambd = params
            ww = invgauss.cdf(sorted_data,mu = mu/lambd, scale = lambd,loc = 0)
        elif dist == 'Gumbel':
            mu, beta = params
            ww = gumbel_r.cdf(sorted_data,mu,beta)
        
        
        plt.figure(figsize=(5,5))
        plt.xlabel('Cuantiles Uniformes [0,1]')
        plt.ylabel('$F(x|\hat{\\mu},\hat{\\lambda})$')
        plt.title('Gráfica de Probabilidad Gaussiana Inversa')
        plt.plot(y_ecdf, Ban1, linestyle='--',linewidth=1,alpha=0.5,color='red')
        plt.plot(y_ecdf, Ban2, linestyle='--',linewidth=1,alpha=0.5,color='red')
        plt.plot([0,1],[0,1],linestyle='-',linewidth=1,alpha=0.5,color='red')
        
        plt.plot(y_ecdf,ww,marker='o', color = "#003f5c",markersize=1.5,linestyle='-',linewidth=0.6)
        plt.show()
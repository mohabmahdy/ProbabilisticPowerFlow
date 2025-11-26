import matplotlib.pyplot as plt
import scipy
import numpy as np

class lognormal_RV():
    """Lognormal distribution with parameters mu, sigma, theta"""
    def __init__(self, mu, sigma, theta, negative = False):
        # For scipy: s = sigma, loc = theta, scale = e^{mu}
        self._mu = mu
        self._sigma = sigma
        self._theta = theta
        self._lognorm = scipy.stats.lognorm(s=sigma, scale=np.exp(mu), loc = 0.0)
        self._lognorm3 = scipy.stats.lognorm(s=sigma, scale=np.exp(mu), loc = theta)
        if negative:
            self._mean = self._theta - self._lognorm.mean()
        else:
            self._mean = self._theta + self._lognorm.mean()
        self._std = self._lognorm.std()
        self._norm = scipy.stats.norm(loc=self._mean, scale=self._std)
        self._negative = negative

    def moment_k(self, k = 1):
        if self._negative:
            raise RuntimeError("Moment not implemented for negative lognormal")
        return self._lognorm3.moment(k)

    def __str__(self):
        if self._negative:
            return f"Negative Lognormal mu={self._mu}, sigma={self._sigma}, theta={self._theta}"
        return f"Lognormal mu={self._mu}, sigma={self._sigma}, theta={self._theta}"

    def __repr__(self):
        if self._negative:
            return f"Negative Lognormal mu={self._mu}, sigma={self._sigma}, theta={self._theta}"
        return f"Lognormal mu={self._mu}, sigma={self._sigma}, theta={self._theta}"

    def pdf(self, x):
        if self._negative:
            return self._lognorm.pdf(self._theta-x)
        return self._lognorm.pdf(x-self._theta)

    def cdf(self, x):
        if self._negative:
            return 1-self._lognorm.cdf(self._theta-x)
        return self._lognorm.cdf(x-self._theta)

    def percentile(self, p = 0.5):
        if self._negative:
            raise RuntimeError("Percentile not implemented for negative lognormal")
        if p >=1:
            raise RuntimeError("The percentile can not be 1 or greater")
        elif p<=0:
            raise RuntimeError("The percentile can not be less than 0")
        return self._lognorm3.ppf(p)

    def get_samples(self, n):
        return self._lognorm3.rvs(size = n).flatten()

    def evaluate_approx(self,samples):
        # Kolmogorov–Smirnov test (max difference between the CDF of the approximation and the CDF of the samples) [0,1]
        return scipy.stats.kstest(samples, self.cdf)

    def cdf_plot(self, samples):
        samples.sort()
        ecdf = np.sort(samples)
        ecdf = np.arange(1, len(ecdf)+1) / len(ecdf)
        xgrid = np.linspace(samples.min(), samples.max(), len(samples))
        plt.plot(xgrid, self.cdf(xgrid), label='CDF fitted lognorm', lw=2)
        plt.plot(samples, ecdf, label='ECDF (empirico)')
        plt.xlabel('x'); plt.ylabel('CDF'); plt.legend(); plt.title('ECDF vs CDF lognormal fitted')
        plt.grid(True)
        plt.show()

    def percentile_plot(self, samples):
        samples.sort()
        ecdf = np.sort(samples)
        ecdf = np.arange(1, len(ecdf)+1) / len(ecdf)
        xgrid = np.linspace(samples.min(), samples.max(), len(samples))

        all_x = []
        all_y = []
        for i in range(len(samples)):
            all_x.append(self.cdf(samples[i]))
            all_y.append(ecdf[i])
        plt.plot(all_x, all_y, label='Percentile')
        plt.plot([0,1], [0,1], label='Perfect Approximation')
        plt.xlabel('CDF of approximation'); plt.ylabel('CDF from data'); plt.legend(); plt.title('Percentile Plot')
        plt.grid(True)
        plt.show()

    def get_mu(self):
        return self._mu
    def get_sigma(self):
        return self._sigma
    def get_theta(self):
        return self._theta
    def get_mean(self):
        return self._mean
    def get_std(self):
        return self._std
import numpy as np
from lognormal import lognormal_RV
def ratio_lognormal(l1:lognormal_RV, l2:lognormal_RV, add_one:bool=False, corr:float = 0.0, v_value:float=1.0)->lognormal_RV:
    if l1 is None:
        return None
    if l2 is None:
        return l1

    if add_one:
        mu_1 = l1.get_mu()
        sigma_1 = l1.get_sigma()
        theta_1 = l1.get_theta()

        mu_2 = l2.get_mu()
        sigma_2 = l2.get_sigma()
        theta_2 = l2.get_theta()

        mean_2 = l2.get_mean()
        v_2 = (l2.get_std())**2

        if v_value <= mean_2:
            raise ValueError("v_value must be greater than the mean value of the losses")

        m_B = np.log(v_value - mean_2) - v_2 / (2 * (v_value - mean_2)**2)
        s_B2 = v_2 / (v_value - mean_2)**2

        mu_lnR = mu_1 - m_B
        sigma_lnR2 = sigma_1**2 + s_B2 + 2*corr*sigma_1*np.sqrt(s_B2)
        
        return lognormal_RV(mu_lnR, np.sqrt(sigma_lnR2), 0)

    else:
        mu_1 = l1.mu
        sigma_1 = l1.sigma
        theta_1 = l1.theta
        mu_2 = l2.mu
        sigma_2 = l2.sigma
        theta_2 = l2.theta

        return lognormal_RV(mu_1-mu_2, np.sqrt(sigma_1**2+sigma_2**2), 0)
    
def sum_lognormal(l1:lognormal_RV, l2:lognormal_RV, weights:list = [1,1], corr:float=0.0, correct_mean:bool=False)->lognormal_RV:
    if l1 is None:
        return l2
    if l2 is None:
        return l1
    

    mean1 = l1.get_mean()
    mean2 = l2.get_mean()

    std1 = l1.get_std()
    std2 = l2.get_std()

    m1 = weights[0]*mean1 + weights[1]*mean2
    m2 = (weights[0]*std1)**2 + (weights[1]*std2)**2 + 2*corr*weights[0]*weights[1]*std1*std2

    if correct_mean:
        m1 = max(1e-10, m1)
        m2 = min(1e12, m2)

    sigma = np.sqrt(np.log(1+m2/(m1**2)))
    mu = np.log(m1) - 0.5*sigma**2

    return lognormal_RV(mu, sigma, 0)

def square_lognormal(l1:lognormal_RV)->lognormal_RV:
    if l1 is None:
        return None

    mu = l1.get_mu()
    sigma = l1.get_sigma()
    theta = l1.get_theta()

    return lognormal_RV(2*mu, 2*sigma, 0)

def power_lognormal(l1:lognormal_RV, a:float = 1.0)->lognormal_RV:
    if l1 is None:
        return None

    mu = l1.get_mu()
    sigma = l1.get_sigma()

    return lognormal_RV(a*mu, a*sigma, 0)

def scalar_lognormal(l1:lognormal_RV, a:float = 1.0)->lognormal_RV:
    if l1 is None:
        return None
    if a <= 0:
        raise ValueError("Scalar must be positive.")
    mu = l1.get_mu()
    sigma = l1.get_sigma()

    return lognormal_RV(mu + np.log(a), sigma, 0)

def diff_lognormal(l1:lognormal_RV, l2:lognormal_RV)->float:
    if l1 is None and l2 is None:
        return 0
    if l1 is None or l2 is None:
        return 1
    return max(abs(l1.get_mu()-l2.get_mu()), abs(l1.get_sigma()-l2.get_sigma()), abs(l1.get_theta()-l2.get_theta()))

def product_lognormal(l1, l2):
    if l1 is None or l2 is None:
        return None

    mu_1 = l1.get_mu()
    sigma_1 = l1.get_sigma()
    theta_1 = l1.get_theta()
    mu_2 = l2.get_mu()
    sigma_2 = l2.get_sigma()
    theta_2 = l2.get_theta()

    return lognormal_RV(mu_1+mu_2, np.sqrt(sigma_1**2+sigma_2**2), 0)







def approx_double_lognormal_to_single(l, theta_val):
    if l is None:
        return None
    mean_val = theta_val
    var_val = 0.0
    val_added = 0
    if l.l_positive is not None:
        mean_val += l.l_positive.get_mean()
        var_val += l.l_positive.get_std()**2
        val_added += 1
    if l.l_negative is not None:
        mean_val -= l.l_negative.get_mean()
        var_val += l.l_negative.get_std()**2
        val_added += 1
        
    if val_added == 0:
        return None
    elif val_added == 2:
        var_val = var_val -2* l.l_negative.get_std() * l.l_positive.get_std()
    
    sigma = np.sqrt(np.log(1+var_val/(mean_val**2)))
    mu = np.log(mean_val) - 0.5*sigma**2
    return lognormal_RV(mu, sigma, 0)

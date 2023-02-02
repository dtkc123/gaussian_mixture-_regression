import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.mixture import GaussianMixture

class GMR(BaseEstimator, RegressorMixin):
    """
    Gaussian mixture regression. Args are comply with sklearn.mixture.GaussianMixture.
    """
    def __init__(self,n_components=1,
                covariance_type='full',
                tol=0.001,
                reg_covar=1e-06,
                max_iter=100,
                n_init=1,
                init_params='kmeans',
                weights_init=None,
                means_init=None,
                precisions_init=None,
                random_state=None,
                warm_start=False,
                verbose=0,
                verbose_interval=10):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components,
                                    covariance_type=covariance_type,
                                    tol=tol,
                                    reg_covar=reg_covar,
                                    max_iter=max_iter,
                                    n_init=n_init,
                                    init_params=init_params,
                                    weights_init=weights_init,
                                    means_init=means_init,
                                    precisions_init=precisions_init,
                                    random_state=random_state,
                                    warm_start=warm_start,
                                    verbose=verbose,
                                    verbose_interval=verbose_interval)
    
    def fit(self, X, Y):
        """
        
        Fitting the GMR model.
        
        Parameters
        ----------
        X : array, shape (n, n_features)
        Y : array, shape (n, n_objectives)

        Returns
        -------
        self
        """
        D = np.hstack([X, Y])
        self.gmm.fit(D)
        self.dim_x = X.shape[1]
        self.dim_y = Y.shape[1]
        self.covariances = np.linalg.inv(self.gmm.precisions_)
        return self
    
    def predict(self, X):
        """
        
        Forward prediction by GMR.
        
        Parameters
        ----------
        X : array, shape (n, n_features)

        Returns
        -------
        Y : array, shape (n, n_objectives)
        """
        weights, covs_xx, covs_xy, covs_yy, means_x, means_y = self._prepare()
        weights_samples = self._calc_weights_each_samples(Y, means_y, covs_yy)
        
        means_y_given_x = 0.
        for i in range(self.n_components):
            means_y_i = means_y[i] + (X - means_x[i].reshape(1,-1)\
                                  ).dot(np.linalg.inv(covs_xx[i])\
                                       ).dot(covs_xy[i])
            means_y_given_x += weights_samples[:,i].reshape(-1,1)*means_y_i
        return means_y_given_x
    
    def _prepare(self):
        means = self.gmm.means_
        weights = self.gmm.weights_
        
        covs_xx = self.covariances[:,:self.dim_x, :self.dim_x]
        covs_xy = self.covariances[:,:self.dim_x, self.dim_x:]
        covs_yy = self.covariances[:,self.dim_x:, self.dim_x:]
        means_x = means[:,:self.dim_x]
        means_y = means[:,self.dim_x:]
        return weights, covs_xx, covs_xy, covs_yy, means_x, means_y
    
    def _calc_weights_each_samples(self, Q, means_q, covs_qq):
        # Q : X or Y
        weights_samples = np.zeros((Q.shape[0], self.n_components))
        for i in range(self.n_components):
            pdf_q_i = multivariate_normal.pdf(Q, mean=means_q[i], cov=covs_qq[i])
            weights_samples[:,i] = self.gmm.weights_[i]*pdf_q_i
        weights_samples = weights_samples/np.sum(weights_samples, axis=1).reshape(-1,1)
        return weights_samples
        
    
    def inverse_predict(self, Y):
        """
        
        Inverse analysis by GMR via mean values of X given Y.
        
        Parameters
        ----------
        Y : array, shape (n, n_objectives)

        Returns
        -------
        X : array, shape (n, n_features)
        """
        weights, covs_xx, covs_xy, covs_yy, means_x, means_y = self._prepare()
        weights_samples = self._calc_weights_each_samples(Y, means_y, covs_yy)
        
        
        means_x_given_y = 0.
        for i in range(self.n_components):
            means_x_i = means_x[i] + (Y - means_y[i].reshape(1,-1)\
                                  ).dot(np.linalg.inv(covs_yy[i])\
                                       ).dot(covs_xy[i].T)
            means_x_given_y += weights_samples[:,i].reshape(-1,1)*means_x_i
        return means_x_given_y
        
    def inverse_sample(self, Y, n_samples = 10000):
        """
        
        Inverse analysis by GMR via sampling many samples for X given Y.
        
        Parameters
        ----------
        Y : array, shape (n, n_objectives)

        Returns
        -------
        X_samples : array, shape (n_samples, n_features)
        """
        if Y.ndim == 1:
            Y = Y.reshape(1,-1)
        weights, covs_xx, covs_xy, covs_yy, means_x, means_y = self._prepare()
        weights_samples = self._calc_weights_each_samples(Y, means_y, covs_yy)
        
        means_x_given_ys = np.zeros([self.n_components, means_x.shape[1]])
        covs_xx_given_ys = np.zeros([self.n_components, means_x.shape[1],means_x.shape[1]])
        for i in range(self.n_components):
            means_x_i = means_x[i] + (Y - means_y[i].reshape(1,-1)\
                                  ).dot(np.linalg.inv(covs_yy[i])\
                                       ).dot(covs_xy[i].T)
            
        
            covs_x_i = covs_xx[i] - covs_xy[i].dot(np.linalg.inv(covs_yy[i]).dot(covs_xy[i].T))
            print(means_x_i.shape)
            print(covs_x_i.shape)
            means_x_given_ys[i] = means_x_i
            covs_xx_given_ys[i] = covs_x_i
        
        X_samples = np.zeros([Y.shape[0], n_samples, means_x.shape[1]])
        for j in range(Y.shape[0]):
            wj = weights_samples[j,:]
            wj /= np.sum(wj)
            
            count = 0
            for i in range(self.n_components):
                if i != self.n_components-1:
                    ns_ij = int(n_samples*wj[i])
                    
                else:
                    ns_ij = n_samples - count
                samples = multivariate_normal.rvs(means_x_given_ys[i,:], covs_xx_given_ys[i,:,:], size=ns_ij)
                X_samples[j,count:ns_ij,:] = samples
                count += ns_ij
        return X_samples
        
if __name__=="__main__":
    X = np.random.randn(100,2)
    X[:50] += 3
    A = np.random.randn(2,3)
    Y = np.dot(X, A) + np.random.randn(100,3)
    gmr = GMR(n_components=2)
    gmr.fit(X[:70],Y[:70])
    Y_pred = gmr.predict(X)
    X_samples = gmr.inverse_sample(Y[0,:],n_samples=10)
    X_pred = gmr.inverse_predict(Y)
    
    
    

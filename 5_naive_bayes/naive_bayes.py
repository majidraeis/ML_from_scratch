import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.means = []
        self.vars = []
        self.priors = []
        for c in self.classes:
            X_c = X[y==c]
            mean_c = np.mean(X_c, axis=0)
            var_c = np.var(X_c, axis=0)
            prior_c = len(X_c)/n_samples
            self.means.append(mean_c)
            self.vars.append(var_c)
            self.priors.append(prior_c)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posterior_list = []
        priors = np.log(self.priors)
        for i, c in enumerate(self.classes):
            mean_c = self.means[i]
            var_c = self.vars[i]
            pdfs = self._pdf(x, mean_c, var_c)
            posterior = np.sum(np.log(pdfs))
            posterior_list.append(posterior)
        log_likelihood = np.array(posterior_list) + priors
        return self.classes[np.argmax(log_likelihood)]

    def _pdf(self, x, mean, var):
        num = np.exp(-((x-mean)**2)/2*var)
        denum = np.sqrt(2*np.pi*var)
        return num/denum


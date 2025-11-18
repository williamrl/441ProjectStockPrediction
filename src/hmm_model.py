"""HMM wrapper: tries hmmlearn, falls back to clustering-based pseudo-HMM.

The fallback estimates hidden states by clustering emissions and computing empirical transitions.
"""
import numpy as np
import logging


class MarketHMM:
    def __init__(self, n_states: int = 2, random_state: int | None = 0):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.fallback = False
        self.transition_matrix_ = None
        self.means_ = None

    def fit(self, X: np.ndarray):
        """Fit an HMM to data X (n_samples, n_features)."""
        # Try hmmlearn
        try:
            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(n_components=self.n_states, covariance_type="full", n_iter=100, random_state=self.random_state)
            self.model.fit(X)
            self.fallback = False
            self.transition_matrix_ = self.model.transmat_
            # approximation of means: model.means_ if available
            try:
                self.means_ = self.model.means_
            except Exception:
                self.means_ = np.zeros((self.n_states, X.shape[1]))
            return self
        except Exception:
            logging.info("hmmlearn not available or failed; using fallback clustering HMM")

        # Fallback: cluster emissions via GaussianMixture or KMeans, then compute transitions
        try:
            from sklearn.mixture import GaussianMixture

            gm = GaussianMixture(n_components=self.n_states, random_state=self.random_state)
            labels = gm.fit_predict(X)
            self.means_ = gm.means_
        except Exception:
            # last resort: KMeans
            try:
                from sklearn.cluster import KMeans

                km = KMeans(n_clusters=self.n_states, random_state=self.random_state)
                labels = km.fit_predict(X)
                # approximate means
                self.means_ = np.vstack([X[labels == k].mean(axis=0) for k in range(self.n_states)])
            except Exception:
                # extremely minimal fallback: random labels
                rng = np.random.RandomState(self.random_state)
                labels = rng.randint(0, self.n_states, size=X.shape[0])
                self.means_ = np.zeros((self.n_states, X.shape[1]))

        # compute empirical transition matrix
        trans = np.zeros((self.n_states, self.n_states), dtype=float)
        for a, b in zip(labels[:-1], labels[1:]):
            trans[a, b] += 1
        # normalize rows
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans = trans / row_sums
        self.transition_matrix_ = trans
        self.fallback = True
        # save labels as last fit result for simple posterior
        self._last_labels = labels
        return self

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Return most likely hidden states for X."""
        if not self.fallback and self.model is not None:
            try:
                return self.model.predict(X)
            except Exception:
                pass
        # fallback: predict cluster labels by assigning to nearest mean
        if getattr(self, "_last_labels", None) is not None and len(self._last_labels) == X.shape[0]:
            return self._last_labels
        # assign by nearest mean
        if self.means_ is None:
            return np.zeros(X.shape[0], dtype=int)
        d = np.linalg.norm(X[:, None, :] - self.means_[None, :, :], axis=2)
        return d.argmin(axis=1)

    def predict_next_state(self, current_state: int) -> int:
        """Return the most probable next state given current_state using transition matrix."""
        if self.transition_matrix_ is None:
            return current_state
        probs = self.transition_matrix_[current_state]
        return int(np.argmax(probs))

    def state_emission_mean(self, state: int):
        if self.means_ is None:
            return None
        return self.means_[state]

    def next_state_probabilities(self, current_state: int) -> np.ndarray:
        """Return the transition probability vector P(s_{t+1}=k | s_t=current_state)."""
        if self.transition_matrix_ is None:
            # uniform fallback
            return np.ones(self.n_states) / float(self.n_states)
        return np.asarray(self.transition_matrix_[current_state], dtype=float)

    def expected_next_return(self, current_state: int, return_index: int = 0) -> float:
        """Compute expected next-day return given current_state.

        - return_index: which emission dimension corresponds to price return (default 0)
        - Uses transition probabilities and per-state emission means.
        """
        probs = self.next_state_probabilities(current_state)
        if self.means_ is None:
            return 0.0
        means = np.asarray(self.means_, dtype=float)
        if means.ndim == 1:
            # single-dim
            em = means
        else:
            em = means[:, return_index]
        return float(np.dot(probs, em))

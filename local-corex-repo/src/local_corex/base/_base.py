"""Shared scaffold for CorEx implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from typing import Any, Dict, Iterable, Optional


class BaseCorex(ABC):
    """Lightweight sklearn-style mixin for CorEx variants."""

    @abstractmethod
    def fit(self, X, y: Optional[Any] = None):  # pragma: no cover - implemented in subclasses
        """Fit the model to data."""

    @abstractmethod
    def transform(self, X):  # pragma: no cover - implemented in subclasses
        """Transform data using the learned representation."""

    def fit_transform(self, X, y: Optional[Any] = None):
        """Default fit + transform pipeline; subclasses can override for efficiency."""
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):  # pragma: no cover - optional override
        """Map latent factors back to input space if supported."""
        raise NotImplementedError("inverse_transform is not implemented for this estimator.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return estimator parameters following sklearn conventions."""
        signature = inspect.signature(self.__init__)
        params: Dict[str, Any] = {}
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if not hasattr(self, name):
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any):
        """Set estimator parameters; ignores unknown keys to match sklearn semantics."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    # Utility helpers -------------------------------------------------
    def _check_positive(self, name: str, value: Any):
        if value is None or value <= 0:
            raise ValueError(f"{name} must be positive; got {value!r}")

    def _set_seed(self, seed):
        """Store seed for downstream use without enforcing RNG strategy."""
        setattr(self, "seed", seed)
        return seed

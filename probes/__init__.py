from probes.base import BaseProbe
from probes.logistic_regression import LogisticRegressionProbe
from probes.mass_mean_shift import MassMeanShiftProbe

PROBE_REGISTRY: dict[str, type[BaseProbe]] = {
    "lr": LogisticRegressionProbe,
    "mms": MassMeanShiftProbe,
}


def get_probe_class(method: str) -> type[BaseProbe]:
    """Look up a probe class by its short name."""
    if method not in PROBE_REGISTRY:
        raise ValueError(
            f"Unknown probe method '{method}'. Available: {list(PROBE_REGISTRY.keys())}"
        )
    return PROBE_REGISTRY[method]

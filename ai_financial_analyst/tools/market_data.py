"""Market benchmark helpers — thin delegating shims to data/market/.

All logic lives in data/market/risk_free.py and data/market/sp500.py.
This module provides backward-compatible function signatures so existing
callers (price_metrics fetch in data/yahoo/metrics.py) work unchanged.
"""

from ..data.market.risk_free import get_risk_free_rate
from ..data.market.sp500 import get_sp500_data

__all__ = ["get_risk_free_rate", "get_sp500_data"]

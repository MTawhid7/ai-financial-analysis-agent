# Backwards-compatibility shim.
# Chart logic lives in ai_financial_analyst/charts/
from ai_financial_analyst.charts import generate_on_demand_chart, generate_all_charts  # noqa: F401
from ai_financial_analyst.charts.pipeline import (  # noqa: F401
    generate_pe_chart,
    generate_metrics_chart,
    generate_radar_chart,
)
from ai_financial_analyst.charts.price_action import (  # noqa: F401
    generate_price_chart,
    generate_candlestick_chart,
)

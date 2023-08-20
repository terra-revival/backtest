import pandas as pd

from .bokeh_utils import (
    PercentTickFormatter,
    create_il_control_figure,
    create_pnl_breakdown_figure,
    create_price_figure,
    create_price_impact_figure,
    create_buy_back_volume_quote_figure,
    create_div_price_compare_figure,
    create_volume_quote_figure,
    create_div_tax_pct_figure,
    create_arb_profit_figure,
    create_reserve_breakdown_figure,
    create_div_tax_quote_figure,
    create_div_exec_price_figure,
    create_trade_price_figure,
    create_trade_volume_figure,
    create_arb_volume_quote_figure,
    create_retail_volume_quote_figure,
    create_portfolio_figure,
)

from bokeh.models import Div

# from .utils import format_df
from .market import MarketPair

default_params={
    "toolbar_location":None,
    "height":200,
}

from bokeh.models import Div
import pandas as pd

from bokeh.models import Div, Panel, Tabs
from bokeh.layouts import column
import pandas as pd


default_params = {
    "toolbar_location": None,
    "height": 200,
}

def default_headline(df_headline: pd.DataFrame, df_asset: pd.DataFrame, df_perf: pd.DataFrame):
    # Create panels for each dataframe
    panel1 = Panel(child=Div(text=format_df(df_headline)), title="Headline")
    panel2 = Panel(child=Div(text=format_df(df_asset)), title="Asset")
    panel3 = Panel(child=Div(text=format_df(df_perf)), title="Performance")

    # Combine panels into tabs
    tabs = Tabs(tabs=[panel1, panel2, panel3])
    return column(tabs)

def format_df(df, width=None):
    html_classes = ["table", "table-striped", "table-hover", "table-primary", "table text-nowrap"]
    if width is None:
        return df.to_html(classes=html_classes)
    else:
        return f'<div style="width: {width}px;">{df.to_html(classes=html_classes)}</div>'


def default_breakdown(df: pd.DataFrame, quote_asset: str):
    default_params_y_label={
        **default_params,
        "y_axis_label":quote_asset,
    }

    return [
        [
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                **default_params,
            ),
            create_price_figure(
                df,
                title="Mid Vs Exec price",
                **default_params_y_label,
            ),
        ],
        [
            create_volume_quote_figure(
                df,
                title=f"Volume (Quote)",
                chart_type="vbar",
                **default_params_y_label,
            ),
            create_price_impact_figure(
                df,
                title="Price Impact",
                **default_params_y_label,
            ),
        ],
        [
            create_arb_profit_figure(
                df,
                title="Arb profit",
                **default_params_y_label,
            ),
            create_il_control_figure(
                df,
                title="IL theoric vs actual",
                idx_column="mkt_price_ratio",
                chart_type="scatter",
                x_axis_formatter=PercentTickFormatter,
                **default_params,
            ),
        ],
        [
            create_arb_volume_quote_figure(
                df,
                title=f"Volume Breakdown (Quote)",
                chart_type="vbar_stack",
                **default_params_y_label,
            ),
            create_portfolio_figure(
                df,
                title=f"Portfolio Value HOLD VS LP (Quote)",
                **default_params_y_label,
            ),
        ],
    ]


def div_layout(df: pd.DataFrame, quote_asset: str):
    default_params={
        "toolbar_location":None,
        "height":200,
    }

    default_params_y_label={
        **default_params,
        "y_axis_label":quote_asset,
    }

    return [
        [
            create_div_price_compare_figure(
                df,
                title="Price Div Tax ON/OFF",
                **default_params_y_label,
            ),
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                **default_params,
            ),
        ],
        [
            create_volume_quote_figure(
                df,
                title=f"Volume (Quote)",
                chart_type="vbar",
                **default_params_y_label,
            ),
            create_il_control_figure(
                df,
                title="IL theoric vs actual",
                idx_column="mkt_price_ratio",
                chart_type="scatter",
                x_axis_formatter=PercentTickFormatter,
                **default_params,
            ),
        ],
        [
            create_div_tax_quote_figure(
                df,
                title="Div Tax",
                chart_type="vbar",
                **default_params,
            ),
            create_price_impact_figure(
                df,
                title="Price Impact",
                **default_params_y_label,
            ),
        ],
        [
            create_buy_back_volume_quote_figure(
                df,
                title="Buy back volume",
                chart_type="vbar",
                **default_params_y_label,
            ),
            create_div_tax_pct_figure(
                df,
                title="Div Tax (%)",
                **default_params,
            ),
        ],
        [
            create_reserve_breakdown_figure(
                df,
                title="Reserve Base/Quote",
                chart_type="vbar_stack",
                **default_params_y_label,
            ),
            create_portfolio_figure(
                df,
                title=f"Portfolio Value HOLD VS LP (Quote)",
                **default_params_y_label,
            ),
        ],
    ]

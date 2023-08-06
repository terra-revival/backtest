import pandas as pd

from .bokeh_utils import (
    create_exec_price_figure,
    create_il_control_figure,
    create_pnl_breakdown_figure,
    create_price_figure,
    create_price_impact_figure,
    create_buy_back_volume_quote_figure,
    create_reserve_account_figure,
    create_div_tax_quote_figure,
    create_div_price_compare_figure,
    create_volume_quote_figure,
    create_div_volume_quote_figure,
    create_div_tax_pct_figure,
)


def default_layout(df: pd.DataFrame):
    return [
        [
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                x_axis_type="datetime",
                y_percent_format=True,
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_price_figure(
                df,
                title="Mid Vs Mkt price",
                x_axis_type="datetime",
                y_percent_format=False,
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_volume_quote_figure(
                df,
                title="Volume (Quote)",
                chart_type="vbar",
                x_axis_type="datetime",
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_pnl_breakdown_figure(
                df,
                title="IL Breakdown",
                idx_column="mkt_price_ratio",
                chart_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_price_impact_figure(
                df,
                title="Price Impact",
                x_axis_type="datetime",
                y_percent_format=True,
                toolbar_location=None,
            ),
            create_il_control_figure(
                df,
                title="IL theoric vs actual",
                idx_column="mkt_price_ratio",
                chart_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                x_desired_num_ticks=4,
                toolbar_location=None,
            ),
            create_exec_price_figure(
                df,
                title="Exec Vs Mkt price",
                chart_type="distribution",
                x_percent_format=False,
                y_percent_format=True,
                toolbar_location=None,
            ),
        ],
    ]


def div_layout(df: pd.DataFrame, include_default_layout=False):
    layout = default_layout(df) if include_default_layout else []
    layout.extend([
        [
            create_div_price_compare_figure(
                df,
                title="Price Div Tax ON/OFF",
                x_axis_type="datetime",
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_div_volume_quote_figure(
                df,
                title="Volume (Quote))",
                chart_type="vbar_stack",
                x_axis_type="datetime",
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_buy_back_volume_quote_figure(
                df,
                title="Buy back volume (Quote)",
                chart_type="vbar",
                x_axis_type="datetime",
                toolbar_location=None,
                height=150,
            ),
        ],
        [
            create_div_tax_quote_figure(
                df,
                title="Div Tax (Quote)",
                chart_type="vbar",
                x_axis_type="datetime",
                toolbar_location=None,
            ),
            create_reserve_account_figure(
                df,
                title="Reserve (Quote)",
                chart_type="vbar",
                x_axis_type="datetime",
                toolbar_location=None,
            ),
        ],
    ])
    return layout

def new_div_layout(df: pd.DataFrame):
    default_axis={
        "x_axis_type":"datetime",
    }
    default_figure={
        "toolbar_location":None,
        "height":150,
    }
    default_params={
        **default_axis,
        **default_figure,
    }

    return [
        [
            create_div_price_compare_figure(
                df,
                title="Price Div Tax ON/OFF",
                toolbar_location=None,
                height=200,
                **default_axis,
            ),
        ],
        [
            create_div_volume_quote_figure(
                df,
                title="Volume (Quote))",
                chart_type="vbar_stack",
                **default_params,
            ),
        ],
        [
            create_reserve_account_figure(
                df,
                title="Reserve (Quote)",
                chart_type="vbar",
                **default_params,
            ),
            create_buy_back_volume_quote_figure(
                df,
                title="Buy back volume (Quote)",
                chart_type="vbar",
                **default_params,
            ),
        ],
        [
            create_div_tax_pct_figure(
                df,
                title="Div Tax (%)",
                y_percent_format=True,
                **default_params,
            ),
            create_price_impact_figure(
                df,
                title="Price Impact",
                y_percent_format=True,
                **default_params,
            ),
        ],
        [
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                y_percent_format=True,
                **default_params,
            ),
            create_pnl_breakdown_figure(
                df,
                title="IL Breakdown",
                idx_column="mkt_price_ratio",
                chart_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                **default_figure,
            ),
        ],
    ]

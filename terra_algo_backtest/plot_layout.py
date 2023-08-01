import pandas as pd

from .bokeh_utils import (
    create_buy_back_price_figure,
    create_div_tax_figure,
    create_div_tax_unused_figure,
    create_exec_price_figure,
    create_il_control_figure,
    create_pnl_breakdown_figure,
    create_price_figure,
    create_price_impact_figure,
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
            ),
            create_pnl_breakdown_figure(
                df,
                title="IL Breakdown",
                idx_column="mkt_price_ratio",
                line_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                toolbar_location=None,
            ),
        ],
        [
            create_price_figure(
                df,
                title="Mid Vs Mkt price",
                x_axis_type="datetime",
                y_percent_format=False,
                toolbar_location=None,
            ),
            create_price_impact_figure(
                df,
                title="Price Impact",
                x_axis_type="datetime",
                y_percent_format=True,
                toolbar_location=None,
            ),
        ],
        [
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                line_type="distribution",
                x_percent_format=True,
                y_percent_format=True,
                toolbar_location=None,
            ),
            create_il_control_figure(
                df,
                title="IL theoric vs actual",
                idx_column="mkt_price_ratio",
                line_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                x_desired_num_ticks=4,
                toolbar_location=None,
            ),
            create_exec_price_figure(
                df,
                title="Exec Vs Mkt price",
                line_type="distribution",
                x_percent_format=False,
                y_percent_format=True,
                toolbar_location=None,
            ),
        ],
    ]


def div_protocol_layout(df: pd.DataFrame):
    return [
        [
            create_buy_back_price_figure(
                df,
                title="Mid Price Buy Back ON/OFF",
                x_axis_type="datetime",
                y_percent_format=False,
                toolbar_location=None,
            ),
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                x_axis_type="datetime",
                y_percent_format=True,
                toolbar_location=None,
            ),
        ],
        [
            create_pnl_breakdown_figure(
                df,
                title="IL Breakdown",
                idx_column="mkt_price_ratio",
                line_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                toolbar_location=None,
            ),
            create_price_impact_figure(
                df,
                title="Price Impact",
                x_axis_type="datetime",
                y_percent_format=True,
                toolbar_location=None,
            ),
        ],
        [
            create_div_tax_figure(
                df,
                title="Div Tax (Quote)",
                x_axis_type="datetime",
                y_percent_format=False,
                toolbar_location=None,
            ),
            create_div_tax_unused_figure(
                df,
                title="Div Tax Excess (Quote)",
                x_axis_type="datetime",
                y_percent_format=False,
                toolbar_location=None,
            ),
            create_pnl_breakdown_figure(
                df,
                title="PnL Breakdown",
                line_type="distribution",
                x_percent_format=True,
                y_percent_format=True,
                toolbar_location=None,
            ),
            create_il_control_figure(
                df,
                title="IL theoric vs actual",
                idx_column="mkt_price_ratio",
                line_type="fitted",
                x_percent_format=True,
                y_percent_format=True,
                x_desired_num_ticks=4,
                toolbar_location=None,
            ),
        ],
    ]

from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import scipy
from bokeh.io import show
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Div, HoverTool, NumeralTickFormatter, Span
from bokeh.plotting import Figure, figure
from bokeh.transform import dodge
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from market import MarketPair, Pool, TradeOrder, split_ticker
from simulation import swap_simulation
from swap import (
    MidPrice,
    constant_product_curve,
    constant_product_swap,
    order_book,
    price_impact_range,
)
from utils import figure_specialization, format_df, resample, timer_func


def new_constant_product_figure(
    mkt: MarketPair,
    x_min: float | None = None,
    x_max: float | None = None,
    num: int | None = None,
    bokeh_figure: Figure | None = None,
    plot_width=900,
    plot_height=600,
):
    """Plots the constant product AMM curve Y = K / X

    Args:
        pool_1 (Pool):
            Liquidity pool 1

        pool_2 (Pool):
            Liquidity pool 2

        k (float, optional):
            Constant product invariant

        x_min (float, optional):
            Start of the range for the x-axis

        x_max (float, optional):
            End of the range for the x-axis

        num (int, optional):
            Number of points to plot

        bokeh_figure (Figure, optional):
            Bokeh figure to plot on

        plot_width (int, optional):
            Width of the plot

        plot_height (int, optional):
            Height of the plot

    """
    p = bokeh_figure or figure(
        title=f"Constant Product AMM Curve for the pair {mkt.ticker}",
        plot_width=plot_width,
        plot_height=plot_height,
    )
    p.xaxis.axis_label = f"Amount {mkt.pool_1.ticker}"
    p.yaxis.axis_label = f"Amount {mkt.pool_2.ticker}"
    x, y = constant_product_curve(mkt, x_min=x_min, x_max=x_max, num=num)
    p.line(x, y, line_width=2, color="navy", alpha=0.6, legend_label="Y=K/X")
    # display current mid price of the pools
    p = with_price_info(
        p,
        MidPrice(f"{mkt.ticker}", mkt.pool_1.balance, mkt.pool_2.balance),
        "Mid Price",
    )
    return p


def new_price_impact_figure(
    mkt: MarketPair,
    order: TradeOrder | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    num: int | None = None,
    precision: float | None = None,
    bokeh_figure: Figure | None = None,
    plot_width=900,
    plot_height=600,
):
    """Plots the constant product AMM curve with price impact range for the oder of size
    dx.

    Args:
        pool_1 (Pool):
            Liquidity pool 1

        pool_2 (Pool):
            Liquidity pool 2

        k (float, optional):
            Constant product invariant

        dx (float, optional):
            The order size

        x_min (float, optional):
            Start of the range for the x-axis

        x_max (float, optional):
            End of the range for the x-axis

        num (int, optional):
            Number of points to plot

        bokeh_figure (Figure, optional):
            Bokeh figure to plot on

        plot_width (int, optional):
            Width of the plot

        plot_height (int, optional):
            Height of the plot

    """
    p = new_constant_product_figure(
        mkt, x_min, x_max, num, bokeh_figure, plot_width, plot_height
    )
    # coomputes price impact range
    price_impact = price_impact_range(mkt, order, precision=precision)
    # plot price impact range
    p.line(
        [price_impact.start.x, price_impact.end.x],
        [price_impact.start.y, price_impact.end.y],
        line_width=20,
        color="red",
        alpha=0.3,
    )
    # add price impact range tooltips
    p = with_price_info(p, price_impact.mid, "Swap Execution Price")
    p = with_price_info(p, price_impact.end, "Mid Price (after swap)")
    return p


def new_order_book_figure(
    mkt: MarketPair,
    x_min: float | None = None,
    x_max: float | None = None,
    num: int | None = None,
    plot_width=900,
    plot_height=600,
):
    """Plots the cumulative quantity at any mid price according to the formula from the
    paper "Order Book Depth and Liquidity Provision in Automated Market Makers. Orders
    under current mid price are on the bid, and those above the ask.

    Args:
        pool_1 (Pool):
            Liquidity pool 1

        pool_2 (Pool):
            Liquidity pool 2

        k (float, optional):
            Constant product invariant

        x_min (float, optional):
            Start of the range for the x-axis

        x_max (float, optional):
            End of the range for the x-axis

        num (int, optional):
            Number of points to plot

        bokeh_figure (Figure, optional):
            Bokeh figure to plot on

        plot_width (int, optional):
            Width of the plot

        plot_height (int, optional):
            Height of the plot

    """
    p = figure(
        title=f"Constant Product AMM Depth for the pair {mkt.ticker}",
        plot_width=plot_width,
        plot_height=plot_height,
    )
    p.xaxis.axis_label = f"{mkt.ticker} Mid Price"
    p.yaxis.axis_label = "Order Size"
    x, mid, q = order_book(mkt, x_min=x_min, x_max=x_max, num=num)
    bid = [q_i if x_i < mkt.pool_1.initial_deposit else 0 for (x_i, q_i) in zip(x, q)]
    ask = [q_i if x_i > mkt.pool_1.initial_deposit else 0 for (x_i, q_i) in zip(x, q)]
    source = ColumnDataSource(data={"mid": mid, "bid": bid, "ask": ask})
    # depth eg. binance style order book
    p.varea_stack(
        ["bid", "ask"],
        x="mid",
        color=("green", "red"),
        source=source,
        alpha=0.4,
        legend_label=["Bid", "Ask"],
    )
    p.x_range.range_padding = 0
    p.y_range.range_padding = 0
    return p


def new_pool_figure(
    pool_1: Pool, pool_2: Pool, steps=None, plot_width=900, plot_height=600
):
    """Plots eveolution of token reserves by steps (time, simulations, blocks etc.).

    Args:
        pool_1 (Pool):
            Liquidity pool 1

        pool_2 (Pool):
            Liquidity pool 2

        steps (list, optional):
            List of steps

        plot_width (int, optional):
            Width of the plot

        plot_height (int, optional):
            Height of the plot

    """
    TOOLTIPS = [
        (f"{pool_1.ticker}", f"@{pool_1.ticker}" + "{0,0.000}"),
        (f"{pool_2.ticker}", f"@{pool_2.ticker}" + "{0,0.000}"),
    ]

    steps = steps if steps else range(len(pool_1.reserves))
    p = figure(
        title="Pool balance history",
        plot_width=plot_width,
        plot_height=plot_height,
        x_range=steps,
        tooltips=TOOLTIPS,
    )
    p.xaxis.axis_label = "Simulation Steps"
    p.yaxis.axis_label = "Reserves"
    p.x_range.range_padding = 0
    p.xgrid.grid_line_color = None
    source = ColumnDataSource(
        data={
            pool_1.ticker: pool_1.reserves,
            pool_2.ticker: pool_2.reserves,
            "steps": steps,
        }
    )
    p.vbar(
        x=dodge("steps", -0.1, range=p.x_range),
        top=pool_1.ticker,
        source=source,
        width=0.2,
        alpha=0.5,
        color="blue",
        legend_label=f"{pool_1.ticker} Pool",
    )
    p.vbar(
        x=dodge("steps", 0.1, range=p.x_range),
        top=pool_2.ticker,
        source=source,
        width=0.2,
        alpha=0.5,
        color="red",
        legend_label=f"{pool_2.ticker} Pool",
    )
    return p


def plot_asset_reserves(
    df_sim, pool_1_ticker, pool_2_ticker, steps, plot_width=900, plot_height=600
):
    TOOLTIPS = [
        (f"{pool_1_ticker}", f"@{pool_1_ticker}" + "{0,0.000}"),
        (f"{pool_2_ticker}", f"@{pool_2_ticker}" + "{0,0.000}"),
    ]

    p = figure(
        title="Pool balance history",
        plot_width=plot_width,
        plot_height=plot_height,
        tooltips=TOOLTIPS,
        x_range=steps,
    )
    p.xaxis.axis_label = ""
    p.yaxis.axis_label = "Reserves"
    p.x_range.range_padding = 0
    p.xgrid.grid_line_color = None
    source = ColumnDataSource(
        data={
            pool_1_ticker: df_sim["asset_quote_pct"].values,
            pool_2_ticker: df_sim["asset_base_pct"].values,
            "steps": steps,
        }
    )
    p.vbar(
        x=dodge("steps", -0.1, range=p.x_range),
        top=pool_1_ticker,
        source=source,
        width=0.2,
        alpha=0.5,
        color="blue",
        legend_label=f"{pool_1_ticker} Pool",
    )
    p.vbar(
        x=dodge("steps", 0.1, range=p.x_range),
        top=pool_2_ticker,
        source=source,
        width=0.2,
        alpha=0.5,
        color="red",
        legend_label=f"{pool_2_ticker} Pool",
    )
    return p


def new_asset_figure(mkt, df_sim, plot_width=900, plot_height=600):
    df_sim["asset_quote_pct"] = 1 - df_sim["asset_base_pct"]
    df_sim_daily, steps = resample(
        df_sim,
        {
            "asset_base_pct": np.mean,
            "asset_quote_pct": np.mean,
        },
    )
    return plot_asset_reserves(
        df_sim_daily,
        mkt.pool_1.ticker,
        mkt.pool_2.ticker,
        steps,
        plot_width=plot_width,
        plot_height=plot_height,
    )


def with_price_info(p, mid: MidPrice, price_label: str) -> figure:
    """Hover tool with price info for the given point."""
    point_id = str(hash(str(mid.x) + str(mid.y)))
    p.circle([mid.x], [mid.y], size=10, color="red", alpha=0.4, name=point_id)
    # use hover tool to display info
    hover = p.select(dict(type=HoverTool, names=[point_id]))
    if not hover:
        hover = HoverTool(names=[point_id])
        p.add_tools(hover)
    hover.tooltips = [
        (f"{mid.x_ticker}", f"{mid.x:.3f}"),
        (f"{mid.y_ticker}", f"{mid.y:.3f}"),
        (f"{price_label}", f"{mid.mid_price:.3f}"),
    ]
    return p


def find_cointegrated_pairs(df1, df2):
    data = np.column_stack((df1["price"].values, df2["price"].values))
    result = coint_johansen(data, det_order=0, k_ar_diff=1)

    trace_stat = result.lr1[0]  # Trace test statistic
    trace_crit_value = result.cvt[0, 0]  # Trace test critical value

    if trace_stat > trace_crit_value:
        cointegrated = True
        conclusion = "The two time series are likely cointegrated."
    else:
        cointegrated = False
        conclusion = "The two time series are likely not cointegrated."

    return {
        "trace_stat": trace_stat,
        "trace_crit_value": trace_crit_value,
        "cointegrated": cointegrated,
        "conclusion": conclusion,
    }


def format_cointegration_result(result):
    trace_stat = result["trace_stat"]
    trace_crit_value = result["trace_crit_value"]
    conclusion = result["conclusion"]

    html = f"""
    <div>
        <h3>Cointegration Test Results:</h3>
        <p>Trace Test Statistic: {trace_stat:.4f}</p>
        <p>Trace Test Critical Value: {trace_crit_value:.4f}</p>
        <p>Conclusion: {conclusion}</p>
    </div>
    """

    return html


@timer_func
def plot_price_ratio(df1, df2, symbol1, symbol2):
    price_ratio = df1["price"].astype(float) / df2["price"].astype(float)
    mean_price_ratio = np.mean(price_ratio)

    p = figure(
        title=f"Price Ratio: {symbol1}/{symbol2}",
        x_axis_label="Time",
        y_axis_label="Price Ratio",
        x_axis_type="datetime",
        output_backend="webgl",
    )
    p.line(df1.index, price_ratio, line_color="navy", alpha=0.6, line_width=2)
    p.line(
        df1.index,
        [mean_price_ratio] * len(df1),
        line_color="red",
        line_dash="dashed",
        line_width=2,
    )
    return p


@timer_func
def plot_scatterplot(df1, df2, symbol1, symbol2, n_points=1000):
    # Calculate the number of data points in the dataset
    n_data = len(df1)
    n_points = min(n_data, n_points)
    # Sample n_points from the dataset
    sample_indices = np.random.choice(n_data, n_points, replace=False)
    # Subset the data using the sampled indices
    df1_sampled = df1.iloc[sample_indices]
    df2_sampled = df2.iloc[sample_indices]
    # Create the scatterplot
    p = figure(
        title=f"Scatterplot: {symbol1} vs {symbol2}",
        x_axis_label=symbol1,
        y_axis_label=symbol2,
        output_backend="webgl",
    )
    p.scatter(
        df1_sampled["price"],
        df2_sampled["price"],
        marker="circle",
        size=5,
        color="navy",
        alpha=0.5,
    )
    return p


@timer_func
def new_trade_figure(df_1, df_2, df_composite, ticker):
    symbol_1, symbol_2 = split_ticker(ticker)
    coint_res = find_cointegrated_pairs(df_1, df_2)
    title_text = (
        f"{symbol_1}/{symbol_2} Trades "
        f"between {df_1.index.min()} "
        f"and {df_1.index.max()}"
    )

    return layout(
        [
            [Div(text=f"<h1 style='text-align:center'>{title_text}</h1>")],
            [Div(text=format_cointegration_result(coint_res))],
            [
                plot_price_ratio(df_1, df_2, symbol_1, symbol_2),
                plot_scatterplot(df_1, df_2, symbol_1, symbol_2),
            ],
            [
                [
                    Div(text=format_df(df_composite.head(10))),
                ],
                [
                    Div(text=format_df(df_1.head(10))),
                    Div(text=format_df(df_2.head(10))),
                ],
            ],
        ],
        sizing_mode="stretch_both",
    )


def new_simulation_figure(mkt, simul, plot_width=900, plot_height=600):
    df_sim = simul["breakdown"]
    title_text = (
        f"{mkt.ticker} Simulation between {df_sim.index.min()} and {df_sim.index.max()}"
    )
    return layout(
        [
            [Div(text=f"<h1 style='text-align:center'>{title_text}</h1>")],
            [
                [Div(text=format_df(simul["headline"], width=plot_width))],
                [Div(text=format_df(mkt.assets(), width=plot_width))],
                [Div(text=format_df(mkt.perf(), width=plot_width))],
            ],
            [
                [
                    new_pnl_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    )
                ],
                [
                    new_portfolio_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    ),
                ],
            ],
            [
                [
                    new_price_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    ),
                ],
                [
                    new_fitted_pnl_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    )
                ],
            ],
            [
                [
                    new_sim_price_impact_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    ),
                ],
                [
                    new_roi_distrib_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    ),
                ],
            ],
            [
                [
                    new_pnl_arb_figure(
                        df_sim, plot_width=plot_width, plot_height=plot_height
                    ),
                ],
                [
                    new_asset_figure(
                        mkt, df_sim, plot_width=plot_width, plot_height=plot_height
                    ),
                ],
            ],
        ],
        sizing_mode="stretch_both",
    )


def new_div_simulation_figure(mkt, simul, label_add="", plot_width=900, plot_height=600):
    df_sim = simul["breakdown"]
    title_text = (
        f"{mkt.ticker} Divergence Tax Simulation between {label_add} {df_sim.index.min()} and {df_sim.index.max()}"
    )
    return layout(
        [
            [Div(text=f"<h1 style='text-align:center'>{title_text}</h1>")],
            [
                [Div(text=format_df(simul["headline"], width=plot_width))],
                [Div(text=format_df(mkt.assets(), width=plot_width))],
                [Div(text=format_df(mkt.perf(), width=plot_width))],
            ],
            [
                [new_tax_profit_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ),
                    new_buy_back_figure(df_sim, plot_width=plot_width,
                                        plot_height=plot_height),
                    new_swap_pool_figure(df_sim, plot_width=plot_width,
                                         plot_height=plot_height),
                    new_staking_pool_figure(df_sim, plot_width=plot_width,
                                            plot_height=plot_height),
                    new_oracle_pool_figure(df_sim, plot_width=plot_width,
                                           plot_height=plot_height),
                    new_community_pool_figure(df_sim, plot_width=plot_width,
                                              plot_height=plot_height),
                ],
                [new_price_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ), new_pnl_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ), new_portfolio_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ), new_fitted_pnl_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ), new_sim_price_impact_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ), new_roi_distrib_figure(
                    df_sim, plot_width=plot_width, plot_height=plot_height
                ), new_asset_figure(
                    mkt, df_sim, plot_width=plot_width, plot_height=plot_height
                )]
            ],
        ],
        sizing_mode="stretch_both",
    )


def new_df_div(df, plot_width=900, plot_height=600):
    return Div(text=format_df(df))
    # return Div(text=format_df(df),width=plot_width, height=plot_height)


def new_curve_figure(
    df_sim: pd.DataFrame,
    title: str = "No Title",
    x_label: str = "No x_label",
    y_label: str = "No y_label",
    y_cols: List[str] = ["No y_cols"],
    colors: List[str] = ["No colors"],
    y_percent_format: bool = False,
    plot_width=900,
    plot_height=600,
):
    p = figure(
        title=title,
        x_axis_type="datetime",
        plot_width=plot_width,
        plot_height=plot_height,
    )
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.formatter.use_scientific = False
    if y_percent_format:
        p.yaxis.formatter = NumeralTickFormatter(format="0.00%")

    for y_i, c_i in zip(y_cols, colors):
        line = p.line(
            df_sim.index,
            df_sim[y_i],
            line_width=1,
            alpha=0.6,
            color=c_i,
            legend_label=f"{y_i}",
        )

        # Add hover tooltip
        hover = HoverTool(
            tooltips=[
                ("Date", "$x{%F}"),
                (f"{y_i}", "$y{0.00}"),
            ],
            formatters={"$x": "datetime"},
            renderers=[line],
        )
        p.add_tools(hover)

    return p


def new_fitted_curve_figure(
    df_sim: pd.DataFrame,
    title: str = "No Title",
    x_label: str = "No x_label",
    y_label: str = "No y_label",
    x_col: str = "No x_col",
    y_cols: List[str] = ["No y_cols"],
    colors: List[str] = ["No colors"],
    line_dash: List[str] = ["No line_dash"],
    y_percent_format: bool = True,
    plot_width: int = 900,
    plot_height: int = 600,
):
    p = figure(
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.yaxis.formatter.use_scientific = False
    if y_percent_format:
        p.yaxis.formatter = NumeralTickFormatter(format="0.00%")

    x_range = np.linspace(df_sim[x_col].min(), df_sim[x_col].max(), 100)
    for y_col, color, dash in zip(y_cols, colors, line_dash):
        # Assuming df_sim is the DataFrame containing your data
        df_sim_agg = df_sim.groupby(x_col)[y_col].mean().reset_index()
        # Fit a second-order polynomial
        coefficients = np.polyfit(df_sim_agg[x_col], df_sim_agg[y_col], 2)
        # Generate the fitted curve
        fitted_curve = np.poly1d(coefficients)
        # plot line
        p.line(
            x_range,
            fitted_curve(x_range),
            line_width=2,
            color=color,
            alpha=0.6,
            line_dash=dash,
            legend_label=f"{y_col}",
        )
    p.add_layout(
        Span(
            location=df_sim[x_col].iloc[-1],
            dimension="height",
            line_color="navy",
            line_dash="solid",
            line_width=0.5,
        )
    )
    #    p.legend.location = "bottom_right"
    return p


def new_distrib_figure(
    df_sim: pd.DataFrame,
    title: str = "No Title",
    x_label: str = "No x_label",
    y_label: str = "No y_label",
    y_cols: list = None,
    colors: list = None,
    line_dashes: list = None,
    plot_width=900,
    plot_height=600,
):
    if y_cols is None:
        y_cols = []

    if colors is None:
        colors = ["blue"] * len(y_cols)

    if line_dashes is None:
        line_dashes = ["solid"] * len(y_cols)

    p = figure(
        title=title,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.xaxis.formatter.use_scientific = False
    p.yaxis.formatter.use_scientific = False

    for col, color, line_dash in zip(y_cols, colors, line_dashes):
        mu, std = df_sim[col].mean(), df_sim[col].std()
        x = np.linspace(mu - 5 * std, mu + 5 * std, 100)
        shape, location = scipy.stats.norm.fit(df_sim[col])
        distrib = scipy.stats.norm.pdf(x, loc=shape, scale=location)
        distrib_normalized = distrib / np.max(distrib)
        line = p.line(
            x, distrib_normalized, line_width=2, line_color=color, line_dash=line_dash
        )
        hover = HoverTool(
            tooltips=[
                ("x", "$x{0.0000}"),
                ("y", "$y{0.0000}"),
            ],
            renderers=[line],
        )
        p.add_tools(hover)

    return p


@timer_func
@figure_specialization(
    title="P&L",
    x_label="",
    y_label="",
    y_cols=["roi"],
    colors=["navy"],
    y_percent_format=True,
)
def new_pnl_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width: int = 900,
    plot_height: int = 600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Price",
    x_label="",
    y_label="",
    y_cols=["price", "avg_price", "mid_price", "mkt_price"],
    colors=["navy", "red", "black", "grey"],
    y_percent_format=False,
)
def new_price_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Portfolio",
    x_label="",
    y_label="",
    y_cols=["hold_portfolio", "current_portfolio"],
    colors=["navy", "red"],
    y_percent_format=False,
)
def new_portfolio_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Price Impact",
    x_label="",
    y_label="",
    y_cols=["price_impact"],
    colors=["navy"],
    y_percent_format=True,
)
def new_sim_price_impact_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="P&L Arbitrage",
    x_label="",
    y_label="",
    y_cols=["total_arb_profit"],
    colors=["navy"],
    y_percent_format=False,
)
def new_pnl_arb_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="CEX Tax Profit",
    x_label="",
    y_label="",
    y_cols=["total_cex_profit", "total_chain_profit"],
    colors=["navy", "red"],
    y_percent_format=False,
)
def new_tax_profit_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Total buy back volume",
    x_label="",
    y_label="",
    y_cols=["total_buy_back_vol"],
    colors=["navy"],
    y_percent_format=False,
)
def new_buy_back_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Swap pool",
    x_label="",
    y_label="",
    y_cols=["total_swap_pool"],
    colors=["navy"],
    y_percent_format=False,
)
def new_swap_pool_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Staking pool",
    x_label="",
    y_label="",
    y_cols=["total_staking_pool"],
    colors=["navy"],
    y_percent_format=False,
)
def new_swap_pool_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Community pool",
    x_label="",
    y_label="",
    y_cols=["total_community_pool"],
    colors=["navy"],
    y_percent_format=False,
)
def new_community_pool_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Oracle pool",
    x_label="",
    y_label="",
    y_cols=["total_oracle_pool"],
    colors=["navy"],
    y_percent_format=False,
)
def new_oracle_pool_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: List[str],
    colors: List[str],
    y_percent_format: bool,
    plot_width=900,
    plot_height=600,
):
    return new_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        y_percent_format=y_percent_format,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="P&L Vs Perf",
    x_label="",
    y_label="",
    x_col="mkt_price_ratio",
    y_cols=["impermanent_loss", "roi", "trade_pnl_pct", "fees_pnl_pct"],
    colors=["red", "green", "red", "green"],
    line_dash=["solid", "solid", "dashed", "dashed"],
)
def new_fitted_pnl_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    x_col: str,
    y_cols: List[str],
    colors: List[str],
    line_dash: List[str],
    plot_width: int = 900,
    plot_height: int = 600,
):
    return new_fitted_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_col=x_col,
        y_cols=y_cols,
        colors=colors,
        line_dash=line_dash,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="P&L Breakdown Vs Perf",
    x_label="",
    y_label="",
    x_col="mkt_price_ratio",
    y_cols=["trade_pnl_pct", "fees_pnl_pct"],
    colors=["red", "green"],
)
def new_fitted_pnl_breakdown_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    x_col: str,
    y_cols: List[str],
    colors: List[str],
    plot_width: int = 900,
    plot_height: int = 600,
):
    return new_fitted_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_col=x_col,
        y_cols=y_cols,
        colors=colors,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Price Impact Vs Perf",
    x_label="",
    y_label="",
    x_col="mkt_price_ratio",
    y_cols=["price_impact"],
    colors=["navy"],
    line_dash=["solid"],
)
def new_fitted_price_impact_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    x_col: str,
    y_cols: List[str],
    colors: List[str],
    line_dash: List[str],
    plot_width: int = 900,
    plot_height: int = 600,
):
    return new_fitted_curve_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        x_col=x_col,
        y_cols=y_cols,
        colors=colors,
        line_dash=line_dash,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Market Price Returns",
    x_label="",
    y_label="",
    y_cols=["mkt_price_ratio"],
    colors=["navy"],
    line_dashes=["solid"],
)
def new_mkt_price_distrib_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: list,
    colors: list,
    line_dashes: list,
    plot_width=900,
    plot_height=600,
):
    return new_distrib_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        line_dashes=line_dashes,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="ROI",
    x_label="",
    y_label="",
    y_cols=["roi", "trade_pnl_pct", "fees_pnl_pct"],
    colors=["navy", "red", "green"],
    line_dashes=["solid", "dashed", "dashed"],
)
def new_roi_distrib_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: list,
    colors: list,
    line_dashes: list,
    plot_width=900,
    plot_height=600,
):
    return new_distrib_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        line_dashes=line_dashes,
        plot_width=plot_width,
        plot_height=plot_height,
    )


@timer_func
@figure_specialization(
    title="Price Impact",
    x_label="",
    y_label="",
    y_cols=["price_impact"],
    colors=["navy"],
    line_dashes=["solid"],
)
def new_price_impact_distrib_figure(
    df_sim: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    y_cols: list,
    colors: list,
    line_dashes: list,
    plot_width=900,
    plot_height=600,
):
    return new_distrib_figure(
        df_sim,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_cols=y_cols,
        colors=colors,
        line_dashes=line_dashes,
        plot_width=plot_width,
        plot_height=plot_height,
    )


def cp_amm_autoviz(
    mkt: MarketPair,
    df_trades: pd.DataFrame | None = None,
    order: TradeOrder | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    num: int | None = None,
    precision: float | None = None,
    plot_width=900,
    plot_height=600,
):
    """Autoviz for liquidity pools.

    Args:
        pool_1 (Pool):
            Liquidity pool 1

        pool_2 (Pool):
            Liquidity pool 2

        k (float, optional):
            Constant product invariant

        dx (float, optional):
            The order size

        x_min (float, optional):
            Start of the range for the x-axis

        x_max (float, optional):
            End of the range for the x-axis

        num (int, optional):
            Number of points to plot

        precision (float, optional):
            Precision at which the invariant is evaluated

        plot_width (int, optional):
            Width of the plot

        plot_height (int, optional):
            Height of the plot

        compact (bool, optional):
            If True, 2 plots per row are displayed
            Else, 1 plot per row is displayed

    """
    order = order or TradeOrder.create_default(mkt)
    p1 = new_constant_product_figure(
        mkt,
        plot_width=plot_width,
        plot_height=plot_height,
        x_min=x_min,
        x_max=x_max,
        num=num,
    )
    p2 = new_price_impact_figure(
        mkt,
        order,
        precision=precision,
        plot_width=plot_width,
        plot_height=plot_height,
        x_min=x_min,
        x_max=x_max,
        num=num,
    )
    swap_mkt = deepcopy(mkt)
    constant_product_swap(swap_mkt, order, precision=precision)
    p3 = new_pool_figure(
        swap_mkt.pool_1,
        swap_mkt.pool_2,
        ["Before Swap", "After Swap"],
        plot_width=plot_width,
        plot_height=plot_height,
    )
    p4 = new_order_book_figure(mkt, plot_width=plot_width, plot_height=plot_height)
    if df_trades is not None:
        simul = swap_simulation(mkt, df_trades, is_arb_enabled=True)
        show(
            new_simulation_figure(
                mkt, simul, plot_width=plot_width, plot_height=plot_height
            )
        )
    show(layout([[p1, p2], [p3, p4]], sizing_mode="stretch_both"))

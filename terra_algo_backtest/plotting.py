from copy import deepcopy

import numpy as np
from bokeh.io import show
from bokeh.layouts import grid, layout
from bokeh.models import ColumnDataSource, Div, HoverTool
from bokeh.plotting import Figure, figure
from bokeh.transform import dodge
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .market import MarketPair, Pool, TradeOrder, split_ticker
from .plot_layout import default_breakdown, default_headline
from .swap import (
    MidPrice,
    constant_product_curve,
    constant_product_swap,
    order_book,
    price_impact_range,
)
from .utils import format_df, resample, timer_func


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


def cp_amm_autoviz(
    mkt: MarketPair,
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
    show(layout([[p1, p2], [p3, p4]], sizing_mode="stretch_both"))


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
            Div(text=f"<h1 style='text-align:center'>{title_text}</h1>"),
        ],
        [
            Div(text=format_cointegration_result(coint_res)),
        ],
        [
            plot_price_ratio(df_1, df_2, symbol_1, symbol_2),
            plot_scatterplot(df_1, df_2, symbol_1, symbol_2),
        ],
        [
            Div(
                text=format_df(
                    df_composite[["price", "quantity", "direction", "asset_unit"]].head(
                        10
                    )
                )
            ),
        ],
        [
            Div(text=format_df(df_1.head(10))),
        ],
        [
            Div(text=format_df(df_2.head(10))),
        ],
        sizing_mode="stretch_both",
    )


def new_df_div(df, plot_width=900, plot_height=600):
    return Div(text=format_df(df))
    # return Div(text=format_df(df),width=plot_width, height=plot_height)


def new_simulation_figure(
    mkt: MarketPair, simul: dict, sim_layout_fn=default_breakdown
) -> layout:
    """Creates a new simulation figure.

    Args:
        mkt (object): The market object.
        simul (dict): The simulation dictionary.

    Returns:
        layout: A Bokeh layout object with the simulation figure.

    """
    df_sim = simul["breakdown"]
    df_sim_stats = simul["headline"]

    sim_layout = sim_layout_fn(df_sim, mkt.pool_1.ticker)
    header_layout = default_headline(df_sim_stats, mkt.assets(), mkt.perf())
    return layout(
        [
            grid(header_layout, sizing_mode="stretch_width"),
        ],
        [
            grid(sim_layout, sizing_mode="stretch_width"),
        ],
        sizing_mode="stretch_width",
    )

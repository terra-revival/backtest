import sys
from functools import partial, update_wrapper
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy
from bokeh.layouts import gridplot, layout
from bokeh.models import ColumnDataSource, Div, HoverTool, NumeralTickFormatter
from bokeh.plotting import Figure, figure
from bokeh.transform import dodge
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .market import MarketPair, Pool, TradeOrder, split_ticker
from .swap import MidPrice, constant_product_curve, order_book, price_impact_range
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


def new_df_div(df, plot_width=900, plot_height=600):
    return Div(text=format_df(df))
    # return Div(text=format_df(df),width=plot_width, height=plot_height)


def create_line_data(df_index: pd.Index, df_col: pd.Series) -> Dict[str, List]:
    """Generates line data from a DataFrame column.

    Args:
        df_index (pd.Index): The DataFrame Index.
        df_col (pd.Series): The DataFrame column.

    Returns:
        Dict[str, List]: A dictionary with line data.

    """
    data = {"x": df_index, "y": df_col}
    return data


def create_distrib_data(
    df_col: pd.Series, color: Optional[str] = "navy", line_dash: Optional[str] = "solid"
) -> Dict[str, List]:
    """Generates distribution data from a DataFrame column.

    Args:
        df_col (pd.Series): The DataFrame column.
        color (str, optional): The color of the line.
        line_dash (str, optional): The style of the line.

    Returns:
        Dict[str, List]: A dictionary with distribution data.

    """
    mu, std = df_col.mean(), df_col.std()
    x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    distrib = scipy.stats.norm.pdf(x, loc=mu, scale=std)
    distrib_normalized = distrib / np.max(distrib)
    data = {
        "x": x,
        "y": distrib_normalized,
        "color": [color] * len(x),
        "line_dash": [line_dash] * len(x),
    }
    return data


def create_fitted_data(
    df_pivot_col: pd.Series, df_col_agg: pd.Series
) -> Dict[str, List]:
    """Generates fitted data from DataFrame columns.

    Args:
        df_pivot_col (pd.Series): The DataFrame pivot column.
        df_col_agg (pd.Series): The DataFrame aggregate column.

    Returns:
        Dict[str, List]: A dictionary with fitted data.

    """
    coefficients = np.polyfit(df_pivot_col, df_col_agg, 2)
    fitted_curve = np.poly1d(coefficients)
    x_range = np.linspace(df_pivot_col.min(), df_pivot_col.max(), 100)
    return {"x": x_range.tolist(), "y": fitted_curve(x_range).tolist()}


def create_data_source(
    df: pd.DataFrame,
    column: str,
    line_type: str,
    idx_column: Optional[str] = None,
) -> Dict[str, List]:
    """Creates a data source based on line type.

    Args:
        df (pd.DataFrame): The DataFrame.
        column (str): The DataFrame column to be processed.
        line_type (str): The line type, choose from 'line', 'distribution', or 'fitted'.
        idx_column (str, optional): The index column of the DataFrame.

    Returns:
        Dict[str, List]: A dictionary with the processed data.

    """
    if line_type == "line":
        return create_line_data(df.index, df[column])
    elif line_type == "distribution":
        return create_distrib_data(df[column])
    elif line_type == "fitted":
        if idx_column is None:
            raise ValueError("idx_column must be specified for fitted line type.")
        df_group = df.groupby(idx_column)[column].mean().reset_index()
        return create_fitted_data(df_group[idx_column], df_group[column])
    else:
        raise ValueError(
            "Invalid line type. Choose from 'line', 'distribution', or 'fitted'"
        )


def create_bokeh_figure(
    data_sets: List[dict], y_percent_format: Optional[bool] = False, **kwargs
) -> figure:
    """Creates a Bokeh figure.

    Args:
        data_sets (List[dict]): A list of data sets for the figure.
        y_percent_format (bool, optional): If True, formats the y-axis as percentages.

    Returns:
        figure: A Bokeh figure.

    """
    p = figure(**kwargs)
    if y_percent_format:
        p.yaxis.formatter = NumeralTickFormatter(format="0.00%")

    for data in data_sets:
        source = ColumnDataSource(data)
        line_figure = p.line(
            x="x",
            y="y",
            line_width=1,
            alpha=0.6,
            color="navy",
            line_dash="solid",
            source=source,
        )
        hover = HoverTool(
            tooltips=[("x", "@x{%F}"), ("y", "@y{0.00}")],
            formatters={"@x": "datetime"},
            renderers=[line_figure],
        )
        p.add_tools(hover)
    return p


def create_line_chart(
    df: pd.DataFrame,
    column: str,
    line_type: str = "line",
    idx_column: Optional[str] = None,
    **kwargs,
) -> figure:
    """Creates a line chart from DataFrame and column.

    Args:
        df (pd.DataFrame): The DataFrame.
        column (str): The DataFrame column to be processed.
        line_type (str): The line type, choose from 'line', 'distribution', or 'fitted'.
        idx_column (str, optional): The index column of the DataFrame.

    Returns:
        figure: A Bokeh figure object with the line chart.

    """
    data = create_data_source(df, column, line_type, idx_column=idx_column)
    p = create_bokeh_figure([data], title=column, **kwargs)
    return p


def register(new_function_name: str, column: str) -> Callable:
    """Registers a new function for a specific column.

    Args:
        new_function_name (str): The name of the new function.
        column (str): The DataFrame column to be processed.

    Returns:
        Callable: The new function.

    """
    new_function = partial(create_line_chart, column=column)
    new_function.__doc__ = f"{new_function_name} wrapper with column={column}."
    update_wrapper(new_function, create_line_chart)
    setattr(sys.modules[__name__], new_function_name, new_function)
    return new_function


# Register new functions
create_mid_price_figure = register("create_mid_price_figure", "mid_price")
create_mkt_price_figure = register("create_mkt_price_figure", "mkt_price")
create_price_impact_figure = register("create_price_impact_figure", "price_impact")
create_pnl_figure = register("create_pnl_figure", "roi")
create_trade_pnl_pct_figure = register("create_trade_pnl_pct_figure", "trade_pnl_pct")
create_fees_pnl_pct_figure = register("create_fees_pnl_pct_figure", "fees_pnl_pct")
create_il_figure = register("create_il_figure", "impermanent_loss")


def create_simulation_gridplot(mkt: object, simul: dict) -> gridplot:
    """Creates a simulation grid plot.

    Args:
        mkt (object): The market object.
        simul (dict): The simulation dictionary.

    Returns:
        gridplot: A Bokeh grid plot.

    """
    df = simul["breakdown"]
    return gridplot(
        [
            [
                create_pnl_figure(df, x_axis_type="datetime"),
                create_trade_pnl_pct_figure(df, x_axis_type="datetime"),
                create_fees_pnl_pct_figure(df, x_axis_type="datetime"),
            ],
            [
                create_price_impact_figure(df, x_axis_type="datetime"),
                create_mid_price_figure(df, x_axis_type="datetime"),
                create_mkt_price_figure(df, x_axis_type="datetime"),
            ],
            [
                create_il_figure(df, idx_column="mkt_price_ratio", line_type="fitted"),
                create_trade_pnl_pct_figure(
                    df, idx_column="mkt_price_ratio", line_type="fitted"
                ),
                create_fees_pnl_pct_figure(
                    df, idx_column="mkt_price_ratio", line_type="fitted"
                ),
            ],
            [
                create_pnl_figure(df, line_type="distribution", y_percent_format=True),
                create_trade_pnl_pct_figure(
                    df, line_type="distribution", y_percent_format=True
                ),
                create_fees_pnl_pct_figure(
                    df, line_type="distribution", y_percent_format=True
                ),
            ],
        ],
        sizing_mode="scale_both",
    )


def new_simulation_figure(mkt: MarketPair, simul: dict) -> layout:
    """Creates a new simulation figure.

    Args:
        mkt (object): The market object.
        simul (dict): The simulation dictionary.

    Returns:
        layout: A Bokeh layout object with the simulation figure.

    """
    df_sim = simul["breakdown"]
    title_text = (
        f"{mkt.ticker} Simulation between {df_sim.index.min()} and {df_sim.index.max()}"
    )
    return layout(
        [
            [Div(text=f"<h1 style='text-align:center'>{title_text}</h1>")],
            [
                [Div(text=format_df(simul["headline"]))],
                [Div(text=format_df(mkt.assets()))],
                [Div(text=format_df(mkt.perf()))],
            ],
            [
                create_simulation_gridplot(mkt, simul),
            ],
        ],
        sizing_mode="scale_both",
    )

import sys
from functools import partial, update_wrapper
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import scipy
from bokeh.models import (
    BasicTicker,
    ColumnDataSource,
    DatetimeTickFormatter,
    HoverTool,
    NumeralTickFormatter,
)
from bokeh.plotting import Figure, figure


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


def create_distrib_data(df_col: pd.Series) -> Dict[str, List]:
    """Generates distribution data from a DataFrame column.

    Args:
        df_col (pd.Series): The DataFrame column.

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


def create_data_sources(
    df: pd.DataFrame,
    columns: List[str],
    line_type: str,
    idx_column: str | None,
) -> List[Dict[str, List]]:
    """Creates a data source based on line type.

    Args:
        df (pd.DataFrame): The DataFrame.
        column (str): The DataFrame column to be processed.
        line_type (str): The line type, choose from 'line', 'distribution', or 'fitted'.
        idx_column (str, optional): The index column of the DataFrame.

    Returns:
        Dict[str, List]: A dictionary with the processed data.

    """
    data_sources = []
    for column in columns:
        if line_type == "line":
            data_sources.append(create_line_data(df.index, df[column]))
        elif line_type == "distribution":
            data_sources.append(create_distrib_data(df[column]))
        elif line_type == "fitted":
            if idx_column is None:
                raise ValueError("idx_column must be specified for fitted line type.")
            df_group = df.groupby(idx_column)[column].mean().reset_index()
            data_sources.append(
                create_fitted_data(df_group[idx_column], df_group[column])
            )
        else:
            raise ValueError(
                "Invalid line type. Choose from 'line', 'distribution', or 'fitted'"
            )
    return data_sources


def get_axis_num_format(data: List[float]) -> str:
    """Automatically determine the precision for NumeralTickFormatter based on the range
    of the data."""

    min_val = min(abs(val) for val in data if val != 0)
    max_val = max(abs(val) for val in data)

    if max_val >= 1e5:
        format_str = "0.0a"
    elif min_val < 1e-3:
        format_str = "0.00000"
    else:
        format_str = "0,0"

    return format_str


def with_axis_format(
    p: Figure,
    data_sets: List[dict],
    y_percent_format: bool,
    x_percent_format: bool,
    x_desired_num_ticks: int,
    y_desired_num_ticks: int,
    **kwargs,
) -> Figure:
    """Sets the axis format for a Bokeh figure.

    Args:
        p (Figure): A Bokeh figure.
        data_sets (List[dict]): A list of data sets for the figure.
        y_percent_format (bool, optional): If True, formats the y-axis as percentages.
        x_percent_format (bool, optional): If True, formats the x-axis as percentages.
        x_axis_type (str, optional): The x-axis type, choose from 'datetime' or None.

    Returns:
        Figure: A Bokeh figure with the axis format set.

    """

    if x_percent_format:
        p.xaxis.formatter = NumeralTickFormatter(format="0.00%")
    elif kwargs.get("x_axis_type", None) == "datetime":
        p.xaxis.formatter = DatetimeTickFormatter(
            years=["%m/%d/%y"],
            months=["%m/%d/%y"],
            days=["%m/%d/%y"],
            hours=["%m/%d/%y %H:%M"],
            minutes=["%m/%d/%y %H:%M"],
            seconds=["%m/%d/%y %H:%M:%S"],
        )
    else:
        p.xaxis.formatter.use_scientific = False
    #        num_format = get_axis_num_format(data_sets[0]["x"])
    #        p.xaxis.formatter = NumeralTickFormatter(format=num_format)

    if y_percent_format:
        p.yaxis.formatter = NumeralTickFormatter(format="0.00%")
    else:
        p.yaxis.formatter.use_scientific = False
    #        num_format = get_axis_num_format(data_sets[0]["y"])
    #        p.yaxis.formatter = NumeralTickFormatter(format=num_format)

    # Set the tickers
    p.xaxis.ticker = BasicTicker(desired_num_ticks=x_desired_num_ticks)
    p.yaxis.ticker = BasicTicker(desired_num_ticks=y_desired_num_ticks)

    return p


def create_bokeh_figure(
    data_sets: List[dict],
    colors: list = ["navy"],
    line_dash: list = ["solid"],
    y_percent_format: bool = False,
    x_percent_format: bool = False,
    x_desired_num_ticks: int = 5,
    y_desired_num_ticks: int = 5,
    **kwargs,
) -> Figure:
    """Creates a Bokeh figure.

    Args:
        data_sets (List[dict]): A list of data sets for the figure.
        y_percent_format (bool, optional): If True, formats the y-axis as percentages.

    Returns:
        figure: A Bokeh figure.

    """
    p = figure(**kwargs)
    p = with_axis_format(
        p,
        data_sets,
        y_percent_format,
        x_percent_format,
        x_desired_num_ticks,
        y_desired_num_ticks,
        **kwargs,
    )
    for i, data in enumerate(data_sets):
        source = ColumnDataSource(data)
        line_figure = p.line(
            x="x",
            y="y",
            line_width=1.5,
            alpha=0.6,
            color=colors[i % len(colors)],
            line_dash=line_dash[i % len(line_dash)],
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
    columns: List[str],
    line_type: str = "line",
    idx_column: str = None,
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
    data_sources = create_data_sources(df, columns, line_type, idx_column=idx_column)
    p = create_bokeh_figure(data_sources, **kwargs)
    return p


def register(new_function_name: str, columns: List[str], **kwargs) -> Callable:
    """Registers a new function for a specific column.

    Args:
        new_function_name (str): The name of the new function.
        column (str): The DataFrame column to be processed.

    Returns:
        Callable: The new function.

    """
    new_function = partial(create_line_chart, columns=columns, **kwargs)
    new_function.__doc__ = (
        f"{new_function_name} wrapper with column={','.join(columns)}."
    )
    update_wrapper(new_function, create_line_chart)
    setattr(sys.modules[__name__], new_function_name, new_function)
    return new_function


# Register create_line_chart functions
create_mid_price_figure = register("create_mid_price_figure", ["mid_price"])
create_mkt_price_figure = register("create_mkt_price_figure", ["mkt_price"])
create_spread_figure = register("create_spread_figure", ["spread"])
create_price_impact_figure = register("create_price_impact_figure", ["price_impact"])

create_pnl_figure = register("create_pnl_figure", ["roi"])
create_trade_pnl_pct_figure = register("create_trade_pnl_pct_figure", ["trade_pnl_pct"])
create_fees_pnl_pct_figure = register("create_fees_pnl_pct_figure", ["fees_pnl_pct"])
create_il_figure = register("create_il_figure", ["impermanent_loss"])

# Register multicurve create_line_chart functions
create_pnl_breakdown_figure = register(
    new_function_name="create_pnl_breakdown_figure",
    columns=["roi", "trade_pnl_pct", "fees_pnl_pct"],
    colors=["navy", "crimson", "green"],
    line_dash=["solid", "dashed", "dashed"],
)

create_il_control_figure = register(
    new_function_name="create_il_control_figure",
    columns=["impermanent_loss", "trade_pnl_pct"],
    colors=["grey", "navy"],
    line_dash=["solid", "dotted"],
)

create_price_figure = register(
    new_function_name="create_price_figure",
    columns=["mkt_price", "mid_price"],
    colors=["grey", "navy"],
    line_dash=["solid", "dotted"],
)

create_exec_price_figure = register(
    new_function_name="create_exec_price_figure",
    columns=["mkt_price", "price"],
    colors=["grey", "navy"],
    line_dash=["solid", "dotted"],
)

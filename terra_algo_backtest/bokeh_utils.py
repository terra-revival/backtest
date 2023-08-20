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
# type alias
TickFormatter = NumeralTickFormatter | DatetimeTickFormatter


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


def create_vbar_data(df_col: pd.Series) -> Dict[str, List]:
    return {
        "x": df_col.index,
        "top": df_col.tolist(),
    }


def create_vbar_stacked_data(df_cols: pd.DataFrame) -> Dict[str, List]:
    data = {"x": df_cols.index.tolist()}
    for col in df_cols.columns:
        data[col] = df_cols[col].tolist()
    return data


def create_data_sources(
    df: pd.DataFrame,
    columns: List[str],
    chart_type: str,
    idx_column: str | None,
    group_method: str = "mean",
) -> List[Dict[str, List]]:
    data_sources = []
    for column in columns:
        if chart_type == "line":
            data_sources.append(create_line_data(df.index, df[column]))
        elif chart_type == "scatter":
            data_sources.append(create_line_data(df[idx_column], df[column]))  # scatter uses same data as line
        elif chart_type == "distribution":
            data_sources.append(create_distrib_data(df[column]))
        elif chart_type == "fitted":
            if idx_column is None:
                raise ValueError("idx_column must be specified for fitted chart type.")

            if group_method == "mean":
                df_group = df.groupby(idx_column)[column].mean().reset_index()
            elif group_method == "sum":
                df_group = df.groupby(idx_column)[column].sum().reset_index()
            elif group_method == "count":
                df_group = df.groupby(idx_column)[column].count().reset_index()
            else:
                raise ValueError("Invalid aggr_method. Choose from 'mean' or 'sum'.")

            data_sources.append(create_fitted_data(df_group[idx_column], df_group[column]))
        elif chart_type == "vbar":
            data_sources.append(create_vbar_data(df[column]))
        elif chart_type == "vbar_stack":
            data_sources.append(create_vbar_stacked_data(df[columns]))
        else:
            raise ValueError(
                "Invalid chart type. Choose from 'line', 'scatter', 'distribution', 'fitted', or 'vbar'"
            )
    return data_sources


def create_bokeh_figure(
    data_sets: List[dict],
    chart_type: str,
    colors: list = ["navy"],
    line_dash: list = ["solid"],
    bar_width: float = 86_400_000, # 1 day
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
    for i, data in enumerate(data_sets):
        source = ColumnDataSource(data)
        color=colors[i % len(colors)]
        if chart_type in ['line', 'scatter', 'distribution', 'fitted']:
            figure_plot = p.line(
                x="x",
                y="y",
                line_width=1.5,
                alpha=0.6,
                color=color,
                line_dash=line_dash[i % len(line_dash)],
                source=source,
            ) if chart_type != 'scatter' else p.scatter(
                x="x",
                y="y",
                alpha=0.6,
                color=color,
                source=source,
            )
            hover = HoverTool(
                tooltips=[("x", "@x{%F}"), ("y", "@y{0.00}")],
                formatters={"@x": "datetime"},
                renderers=[figure_plot],
            )
            p.add_tools(hover)
        elif chart_type == 'vbar':
            p.vbar(
                x="x",
                top="top",
                source=source,
                width=bar_width,
                alpha=0.7,
                color=color,
                line_color="white",
            )
        elif chart_type == 'vbar_stack':
            col_names = data.keys()-["x"]
            p.vbar_stack(
                col_names,
                x="x",
                source=source,
                width=bar_width,
                alpha=0.7,
                line_color="white",
                color=("navy", "red"),
            )


    return p


def create_chart(
    df: pd.DataFrame,
    columns: List[str],
    chart_type: str = "line",
    idx_column: str = None,
    group_method: str = "mean",
    x_desired_num_ticks: int=4,
    y_desired_num_ticks: int=4,
    x_axis_formatter: TickFormatter = None,
    y_axis_formatter: TickFormatter = None,
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
    data_sources = create_data_sources(df, columns, chart_type, idx_column=idx_column, group_method=group_method)
    return with_axis_format(
        p=create_bokeh_figure(data_sources, chart_type, **kwargs),
        x_axis_formatter=x_axis_formatter,
        y_axis_formatter=y_axis_formatter,
        x_desired_num_ticks=x_desired_num_ticks,
        y_desired_num_ticks=y_desired_num_ticks,
    )


def register(new_function_name: str, columns: List[str], **kwargs) -> Callable:
    """Registers a new function for a specific column.

    Args:
        new_function_name (str): The name of the new function.
        column (str): The DataFrame column to be processed.

    Returns:
        Callable: The new function.

    """
    new_function = partial(create_chart, columns=columns, **kwargs)
    new_function.__doc__ = (
        f"{new_function_name} wrapper with column={','.join(columns)}."
    )
    update_wrapper(new_function, create_chart)
    setattr(sys.modules[__name__], new_function_name, new_function)
    return new_function


def with_axis_format(
    p: Figure,
    x_desired_num_ticks: int,
    y_desired_num_ticks: int,
    x_axis_formatter: TickFormatter = None,
    y_axis_formatter: TickFormatter = None,
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
    data = p.renderers[0].data_source.data

    # Infer x_axis formatter if not provided
    if x_axis_formatter is None:
        x_axis_formatter = infer_formatter_type_from_data(data["x"])

    # Infer y_axis formatter if not provided
    if y_axis_formatter is None:
        axis = list(data.keys())
        y_axis = axis[1] if axis[0] == "x" else axis[0]
        y_axis_formatter = infer_formatter_type_from_data(data[y_axis])

    p.xaxis.formatter = x_axis_formatter
    p.yaxis.formatter = y_axis_formatter
    p.xaxis.ticker = BasicTicker(desired_num_ticks=x_desired_num_ticks)
    p.yaxis.ticker = BasicTicker(desired_num_ticks=y_desired_num_ticks)

    return p

DefaultTickFormatter = NumeralTickFormatter(format="0.0a")
PercentTickFormatter = NumeralTickFormatter(format="0.0%")
DecimalTickFormatter = NumeralTickFormatter(format="0.00")
SmallDecimalTickFormatter = NumeralTickFormatter(format="0.0000")
DateTickFormatter = DatetimeTickFormatter(
    years=["%m/%d/%y"],
    months=["%m/%d/%y"],
    days=["%m/%d/%y"],
    hours=["%m/%d/%y %H:%M"],
    minutes=["%m/%d/%y %H:%M"],
    seconds=["%m/%d/%y %H:%M:%S"],
)

def infer_formatter_type_from_data(data):
    """
    Infers the appropriate formatter type based on the provided data.

    Args:
        data (list or array-like): Data for which the formatter needs to be inferred.

    Returns:
        str: Type of the inferred formatter.
    """

    # Check for datetime data
    if isinstance(data[0], (pd.Timestamp, np.datetime64)):
        return DateTickFormatter

    # Check for small numbers (less than 0.01)
    if all(0 < abs(value) < 0.01 for value in data):
        return SmallDecimalTickFormatter

    # Check for small numbers (less than 0.01)
    if all(0 < abs(value) < 0.1 for value in data):
        return DecimalTickFormatter

    # Default case
    return DefaultTickFormatter


# Register create_chart functions
create_trade_price_figure = register("create_trade_price_figure", ["price"])
create_trade_volume_figure = register("create_trade_volume_figure", ["quantity"])

create_mid_price_figure = register("create_mid_price_figure", ["mid_price"])
create_mkt_price_figure = register("create_mkt_price_figure", ["mkt_price"])
create_volume_quote_figure = register("create_volume_quote_figure", ["volume_quote"])

create_spread_figure = register(
    new_function_name="create_spread_figure",
    columns=["spread"],
    y_axis_formatter=NumeralTickFormatter(format="0.000%"),
)

create_pnl_figure = register("create_pnl_figure", ["roi"])
create_trade_pnl_pct_figure = register("create_trade_pnl_pct_figure", ["trade_pnl_pct"])
create_fees_pnl_pct_figure = register("create_fees_pnl_pct_figure", ["fees_pnl_pct"])
create_il_figure = register("create_il_figure", ["impermanent_loss"])

create_arb_profit_figure = register("create_arb_profit_figure", ["arb_profit"])
create_retail_volume_quote_figure = register("create_retail_volume_quote_figure", ["retail_volume_quote"])
create_arb_volume_quote_figure = register(
    new_function_name="create_arb_volume_quote_figure",
    columns=["arb_volume_quote", "retail_volume_quote"],
)

create_portfolio_figure = register(
    new_function_name="create_portfolio_figure",
    columns=["current_portfolio", "hold_portfolio"],
    colors=["navy", "crimson"],
    line_dash=["solid", "dashed"],
)

create_price_impact_figure = register(
    new_function_name="create_price_impact_figure",
    columns=["price_impact"],
    y_axis_formatter=NumeralTickFormatter(format="0.0%"),
)

# Register multicurve create_line_chart functions
create_pnl_breakdown_figure = register(
    new_function_name="create_pnl_breakdown_figure",
    columns=["roi", "trade_pnl_pct", "fees_pnl_pct"],
    colors=["navy", "crimson", "green"],
    line_dash=["solid", "dashed", "dashed"],
    y_axis_formatter=NumeralTickFormatter(format="0.0%"),
)

create_il_control_figure = register(
    new_function_name="create_il_control_figure",
    columns=["impermanent_loss", "trade_pnl_pct"],
    colors=["grey", "navy"],
    line_dash=["solid", "dotted"],
    y_axis_formatter=NumeralTickFormatter(format="0.0%"),
)

create_price_figure = register(
    new_function_name="create_price_figure",
    columns=["mid_price", "mkt_price", "avg_price"],
    colors=["navy", "crimson", "black"],
    line_dash=["solid", "dotted", "dotted"],
)

create_exec_price_figure = register(
    new_function_name="create_exec_price_figure",
    columns=["mkt_price", "price"],
    colors=["grey", "navy"],
    line_dash=["solid", "dotted"],
)

create_div_exec_price_figure = register("create_div_exec_price_figure", ["div_exec_price"])
create_div_tax_quote_figure = register("create_div_tax_quote_figure", ["div_tax_quote"])
create_reserve_account_figure = register("create_reserve_account_figure", ["reserve_account"])
create_buy_back_volume_quote_figure = register("create_buy_back_volume_quote_figure", ["buy_back_volume_quote"])

create_reserve_breakdown_figure = register(
    new_function_name="create_reserve_breakdown_figure",
    columns=["reserve_base_quote", "reserve_quote"],
)

create_reserve_base_figure = register(
    new_function_name="create_reserve_base_figure",
    columns=["reserve_base"],
)

create_reserve_quote_figure = register(
    new_function_name="create_reserve_quote_figure",
    columns=["reserve_quote"],
)

create_div_tax_pct_figure = register(
    new_function_name="create_div_tax_pct_figure",
    columns=["div_tax_pct"],
    y_axis_formatter=NumeralTickFormatter(format="0.0%"),
)

create_div_volume_quote_figure = register(
    new_function_name="create_div_volume_quote_figure",
    columns=["volume_quote", "div_volume_quote"],
)

create_div_price_compare_figure = register(
    new_function_name="create_div_price_compare_figure",
    columns=["mid_price", "no_div_mid_price"],
    colors=["navy", "red"],
    line_dash=["solid", "dotted"],
)

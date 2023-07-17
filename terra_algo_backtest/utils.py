import functools
from hashlib import sha256
from time import time

import matplotlib.pyplot as plt
import pandas as pd
from pandas.util import hash_pandas_object


class CacheDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(self, index=True).values)
        hash_value = hash(hash_value.hexdigest())
        return hash_value

    def __eq__(self, other):
        return self.equals(other)


def timer_func(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def resample(df, agg):
    cols = agg.keys()
    frequencies = ["1Min", "1H", "D", "W", "M", "Y"]
    df_resample = None
    date_format = get_date_format("1Min")

    for freq in frequencies:
        df_resample = df[cols].resample(freq).agg(agg).dropna()

        # Check if the index values are unique and if the length is <= 50
        if df_resample.index.is_unique and len(df_resample) <= 50:
            date_format = get_date_format(freq)
            break

    return df_resample, df_resample.index.strftime(date_format).values


def get_date_format(freq):
    format_map = {
        "1Min": "%m/%d/%Y %H:%M",
        "1H": "%m/%d/%Y %H:%M",
        "D": "%m/%d/%Y",
        "W": "%Y-%m-%d",
        "M": "%Y-%m",
        "Y": "%Y",
    }
    return format_map.get(freq, "%m/%d/%Y")


def format_df(df, width=None):
    if width is None:
        return df.to_html(
            classes=[
                "table",
                "table-striped",
                "table-hover",
                "table-primary",
                "table text-nowrap",
            ]
        )
    else:
        html_classes = ["table", "table-striped", "table-hover", "table-primary"]
        return (
            f'<div style="width: {width}px;">{df.to_html(classes=html_classes)}</div>'
        )


def figure_specialization(**metadata):
    def decorator_figure_specialization(func):
        @functools.wraps(func)
        def wrapper_new_figure(*args, **kwargs):
            # Update the kwargs with the metadata
            kwargs.update(metadata)
            return func(*args, **kwargs)

        return wrapper_new_figure

    return decorator_figure_specialization


def pltShowWaitKey():
    plt.get_current_fig_manager().full_screen_toggle()
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()

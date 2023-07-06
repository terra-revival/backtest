=====
Usage
=====

To use terra_algo_backtest in a project::

    import terra_algo_backtest

Create pools::

    from terra_algo_backtest import Pool

    # liquidity pool made up of reserves of Token A
    pool_token_A = Pool("A", 100)
    # liquidity pool made up of reserves of Token B
    pool_token_B = Pool("B", 100)

Swap::

    from terra_algo_backtest.swap import constant_product_swap

    # order size
    dx = 10
    # swap 10 tokens A for dy tokens B with execution price p
    dy, p = constant_product_swap(dx, pool_token_A, pool_token_B, k)

Tutorials
---------

    * How Uniswap works
    * How Terra Market Module works

See docs/examples folder for interactive version using notebooks

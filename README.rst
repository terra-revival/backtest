=======
terra_algo_backtest
=======


.. image:: https://img.shields.io/pypi/v/terra_algo_backtest.svg
        :target: https://pypi.python.org/pypi/terra_algo_backtest

.. image:: https://img.shields.io/travis/terra-revival/terra_algo_backtest.svg
        :target: https://travis-ci.com/terra-revival/terra_algo_backtest

.. image:: https://readthedocs.org/projects/terra_algo_backtest/badge/?version=latest
        :target: https://terra_algo_backtest.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/terra-revival/terra_algo_backtest/shield.svg
     :target: https://pyup.io/repos/github/terra-revival/terra_algo_backtest/
     :alt: Updates



A package for quantitative analysis and easy data visualisation of constant product automated market makers (CP AMMs)


* Free software: Apache Software License 2.0
* Documentation: https://terra_algo_backtest.readthedocs.io.


Features
--------

* Swap against liquidity pools
* Swap price calculation
* Data visualisation
* Simulation

.. code-block:: python

    from terra_algo_backtest.swap import init_liquidity, MarketQuote, constant_product_swap
    from terra_algo_backtest.plotting import cp_amm_autoviz

    # USDT/USD market price
    usdt_usd = MarketQuote("USDT/USD", 1)
    # UNI/USD market price
    uni_usd = MarketQuote("UNI/USD", 6.32)
    # initialize 2 pools with 10000 USD
    usdt_pool, uni_pool, k = init_liquidity(10000, usdt_usd, uni_usd)
    # plotting data
    cp_amm_autoviz(usdt_pool, uni_pool, compact=True)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

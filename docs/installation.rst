.. highlight:: shell

============
Installation
============


From sources
------------

The sources for terra_algo_backtest can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/terra-revival/terra_algo_backtest

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/terra-revival/terra_algo_backtest/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -r requirements_dev.txt
    $ make dist install


.. _Github repo: https://github.com/terra-revival/terra_algo_backtest
.. _tarball: https://github.com/terra-revival/terra_algo_backtest/tarball/master

Install Jupyter Notebook:

.. code-block:: console

    $ pip install notebook
    # for virtual envs only
    $ python -m ipykernel install --user --name=my_virtual_env

Check out the examples:

.. code-block:: console

    $ cd ./docs/examples
    $ jupyter notebook

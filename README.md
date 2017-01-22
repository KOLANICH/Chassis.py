Chassis.py [![Unlicensed work](https://raw.githubusercontent.com/unlicense/unlicense.org/master/static/favicon.png)](https://unlicense.org/)
===============
![GitLab Build Status](https://gitlab.com/KOLANICH1/Chassis.py/badges/master/pipeline.svg)
[![TravisCI Build Status](https://travis-ci.org/KOLANICH/Chassis.py.svg?branch=master)](https://travis-ci.org/KOLANICH/Chassis.py)
![GitLab Coverage](https://gitlab.com/KOLANICH1/Chassis.py/badges/master/coverage.svg)
[![Coveralls Coverage](https://img.shields.io/coveralls/KOLANICH/Chassis.py.svg)](https://coveralls.io/r/KOLANICH/Chassis.py)
[![Libraries.io Status](https://img.shields.io/librariesio/github/KOLANICH/Chassis.py.svg)](https://libraries.io/github/KOLANICH/Chassis.py)
[![Gitter.im](https://badges.gitter.im/Chassis.py/Lobby.svg)](https://gitter.im/Chassis.py/Lobby)

This is the library to transform a `pandas.DataFrame` into another `DataFrame` suitable for machine learning. It's my own reinvention of a ~~~wheel~~~[```patsy```](https://github.com/pydata/patsy) ![Licence](https://img.shields.io/github/license/pydata/patsy.svg) [![PyPi Status](https://img.shields.io/pypi/v/patsy.svg)](https://pypi.python.org/pypi/patsy)
[![TravisCI Build Status](https://travis-ci.org/pydata/patsy.svg?branch=master)](https://travis-ci.org/pydata/patsy)
[![Coveralls Coverage](https://img.shields.io/coveralls/pydata/patsy.svg)](https://coveralls.io/r/pydata/patsy)
[![Libraries.io Status](https://img.shields.io/librariesio/github/pydata/patsy.svg)](https://libraries.io/github/pydata/patsy)
[![Gitter.im](https://badges.gitter.im/patsy/Lobby.svg)](https://gitter.im/patsy/Lobby) , which doesn't fit my needs.

It solves the following drawbacks of patsy:
* unpredictability
 * the column names are changed in unpredictable way depending on **content** of dataframe you pass to it. You also cannot retrive the names and have to write very dirty code. Here you can retrieve columns by names.
 * The content is often shit `patsy` decides that we need it. For example it can remove a column if it finds them linearry dependent. Such matrices are not suitable to all the ML algorithms and currently there is no way to disable such a behavior.
* lack of automation - I have to do everything myself: construct expression and evaluate it.

Requirements
------------
* [```Python >=3.4```](https://www.python.org/downloads/). [```Python 2``` is dead, stop raping its corpse.](https://python3statement.org/) Use ```2to3``` with manual postprocessing to migrate incompatible code to ```3```. It shouldn't take so much time. For unit-testing you need Python 3.6+ or PyPy3 because their ```dict``` is ordered and deterministic. Python 3 is also semi-dead, 3.7 is the last minor release in 3.
* [```numpy```](https://github.com/numpy/numpy) ![Licence](https://img.shields.io/github/license/numpy/numpy.svg) [![PyPi Status](https://img.shields.io/pypi/v/numpy.svg)](https://pypi.python.org/pypi/numpy) [![TravisCI Build Status](https://travis-ci.org/numpy/numpy.svg?branch=master)](https://travis-ci.org/numpy/numpy) [![Libraries.io Status](https://img.shields.io/librariesio/github/numpy/numpy.svg)](https://libraries.io/github/numpy/numpy)
* [```pandas```](https://github.com/pandas-dev/pandas) ![Licence](https://img.shields.io/github/license/pandas-dev/pandas.svg) [![PyPi Status](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.python.org/pypi/pandas) [![TravisCI Build Status](https://travis-ci.org/pandas-dev/pandas.svg?branch=master)](https://travis-ci.org/pandas-dev/pandas) [![CodeCov Coverage](https://codecov.io/github/pandas-dev/pandas/coverage.svg?branch=master)](https://codecov.io/github/pandas-dev/pandas/) [![Libraries.io Status](https://img.shields.io/librariesio/github/pandas-dev/pandas.svg)](https://libraries.io/github/pandas-dev/pandas) [![Gitter.im](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pydata/pandas)


Tutorial
--------
See [Tutorial.ipynb](./Tutorial.ipynb).


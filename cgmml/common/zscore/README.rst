Z score
=======

This libray is Used for measuring Z score of Children (0-5 Years) based on standard provided by WHO 2006

.. image:: https://img.shields.io/pypi/v/cgmzscore.svg
    :target: https://pypi.org/project/cgmzscore/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/wheel/cgmzscore.svg
    :target: https://pypi.org/project/cgmzscore/

.. image:: https://img.shields.io/pypi/pyversions/cgmzscore.svg
    :target: https://pypi.org/project/cgmzscore/
    
.. image:: https://snyk.io/advisor/python/banner/badge.svg
    :target: https://snyk.io/advisor/python/banner
    :alt: banner

REQUIREMENTS
============

* Python 2.7.x, Python 3.x or later

INSTALLATION
============
`pip install cgmzscore`

EXAMPLE USAGE
=============

calculate z score for weight vs age

.. code-block:: python

    from cgmzscore.src.main import z_score_wfa

    v = z_score_wfa(weight="7.853",age_in_days='16',sex='M')

calculate z score for weight vs length/height

.. code-block:: python

    from cgmzscore.src.main import z_score_wfl

    v = z_score_wfl(weight="7.853",age_in_days='16',sex='M',height='73')

calculate z score for weight vs length/height and both wfl and wfh works same

.. code-block:: python

    from cgmzscore.src.main import z_score_wfh

    v = Calculator().zScore_wfh(weight="7.853",age_in_days='16',sex='M',height='73')

calculate z score for length vs age

.. code-block:: python

    from cgmzscore.src.main import z_score_lhfa

    v = Calculator().z_score_lhfa(age_in_days='16',sex='M',height='73')


calculate all three z score along with class

.. code-block:: python

    from cgmzscore.src.main import z_score_with_class

    v = calculator.z_score_with_class(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

find child is SAM/MAM/Healthy

.. code-block:: python

    from cgmzscore.src.main import calculate_sam_mam

    v = calculate_sam_mam(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

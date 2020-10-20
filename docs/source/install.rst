.. _install:

.. title:: Install

Install
=======

Install on system
-----------------
The first solution to install prevision-quantum-nn is to directly install it on your system:

.. code-block:: bash

    pip install prevision-quantum-nn

.. warning::
        If you have tensorflow-quantum installed on your machine, it is likely that you encounter a problem with the
        sympy version, which is incompatible between strawberryfields (requires thewalrus which requires sympy>=1.5.1). and tensorflow-quantum (requires sympy 1.5).

        Refer to the Dockerfile section if you want to avoid this problem.

Install with Dockerfile
-----------------------

In order to deploy prevision-quantum-nn quickly on any platform, we provide with a simple ``Dockerfile``.

First, build the image with ``docker build``:

.. code-block:: bash

   docker build -t prevision-quantum-nn-image .

Then, run the image

.. code-block:: bash

   docker run -it prevision-quantum-nn-image bash

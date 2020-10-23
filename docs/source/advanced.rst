.. _advanced:

.. title:: Advanced tutorials

==================
Advanced tutorials
==================

Applications described in the Quickstart section, are built over four packages:

1. ``models``
2. ``preprocessing``
3. ``postprocessing``
4. ``dataset``

Tuning your Model
=================

The core of the application is the **model**. Default parameters allow you to start quickly with a model, but it is likely
that you want to change the parameters of the model depending on your needs.
In order to do so, prepare a dictionary with the input parameters and them feed the application with the
``model_params`` option:

.. code-block:: python

        import prevision_quantum_nn as qnn

        model_params = {
            # here, define your model parameters
            # ...
        }
        application = qnn.get_application("classification", model_params=model_params)

Now, let's have an exhaustive description of the different parameters you can tune and how!

Architecture type
-----------------

In order to generate a model that runs on a qubit-based quantum computer (like superconducting qubits, rydberg atoms or
trapped ions), set the parameter ``architecture`` to ``qubit``:

.. code-block:: python

        "architecture": "qubit"

In order to generate a model that runs on a Continuous Variable based quantum computer (like those of Xanadu), set the
parameter ``architecture`` to ``cv``:

.. code-block:: python

        "architecture": "cv"

Number of Qubits/Qumodes
------------------------

Because you may want to simulate a particular hardware, you might need to set the width of the circuit - i.e, the number
of qubits/qumodes.
Set the number of qubits or qumodes used by the model with the ``num_q`` parameter.

.. code-block:: python

        "num_q": 4

.. warning:: 
        Continuous Variable architectures simulations are heavy in memory. If you encounter a processed killed for this
        reason, try diminishing the "cutoff_dim" parameter of the model.

.. code-block:: python

        "cutoff_dim": 6

The interface
-------------
When using pennylane's models, you can choose the interface that will be used to compute the gradient. It can be of two
different types in prevision-quantum-nn: ``"tf"``, which stands for tensorflow, or ``"autograd"``. The default behavior
is ``"autograd"``. You can switch to tensorflow using:

.. code-block:: python
    
    "interface": "tf"

.. warning::
        For now, continuous variable calculations are only restricted to autograd, because we are impatient to see the
        strawberryfields.tf backend available in the stable version of strawberryfields :).

Now, you have all the tools to prepare your quantum hardware architecture! Let's dive into the model that will be
simulated for this hardware. The first thing to do is to encode your data into a proper quantum state.

Encoding classical data into a quantum state
--------------------------------------------

The encoding option depends on the architecture of your hardware.
It is specified by the keyword ``encoding`` in ``model_params``:

For qubit based architectures based on pennylane, there are three different types of encoding allowed.

.. code-block:: python

    "encoding": "angle"

1. ``"angle"`` (default)

2. ``"amplitude"``

3. ``"mottonnen"``

``"amplitude"`` and ``"mottonnen"`` encodes the data into :math:`2^n` amplitudes where :math:`n` is the number of qubits.
Whereas ``"angle"`` encodes data with qubit :math:`n` rotations. More information on this encoding is provided in the
preprocessing section, as your input data must match the quantum encoding.

.. tip::
        From benchmarks results, it seems that the ``"amplitude"`` encoding perfoms worse compared to angle encoding. Try
        angle encoding first, even if you need to reduce the dimension of your features, it might be a better option
        than encoding thousands of features with amplitude encoding. Tools to perform dimension reduction are described
        in the preprocessing section.

For Continuous Variable based architectures based on pennylane, there are two different types of encoding.

.. code-block:: python

    "encoding": "displacement"

1. ``"displacement"`` (default)

2. ``"squeezing"``

Both continuous variable encodings require as many qumodes as the number of features used.

.. tip::
        From benchmarks results, it seems that the ``"displacement"`` encoding perfoms better compared to ``"squeezing"`` encoding.

Building the layers of the circuit
----------------------------------

The layers of model are automatically generated using the keywork ``num_layers``.
In this example, we set it to ``3``.
Also, the keyword ``"layer_type"`` allows to switch from two types of layers.
Use ``template`` if you wish to use pennylane's template type or ``custom`` if you wish to use the one defined by
prevision.io (also default from pennylane examples).

.. code-block:: python

    "num_layers": 3,
    "layer_type": "custom"

.. note::
        If your circuit has only one qubit, the custom layer type will be used automatically. There is no implementation
        of one qubit template layers in pennylane.

.. danger:: 
        The number of layers might be a tricky parameter to tune. For performance considerations, try not to increase it too much, or you will certainly
        fall into the well identified Barren plateau problem.

Now, you have all the tools to set up your hardware architecture to match your data. Let's see how the circuit will be
optimized in order to fit your data!

The optimizer
-------------
Pennylane offers multiple built in optimizers. We have reatined four o them that are available with both tensorflow and
autograd interfaces.

1. ``"SGD"``
2. ``"Adagrad``
3. ``"Adam"``
4. ``"RMSProp"``

The default ``"optimizer_name"`` is ``"Adam"``. Change it to ``"Adagrad"`` with:

.. code-block:: python

    "optimizer_name": "Adagrad"

Once the optimizer name is customized, you also have the choice to tune your ``"learning_rate"`` hyper parameter, or the
``"batch_size"`` used to broadcast you training data set to the optimizer:

.. code-block:: python

    "learning_rate": 0.05,
    "batch_size": 5

The optimizer will run for ``"max_iterations"``, wchi by default is set to 10000.
You can set it to a different value with:

.. code-block:: python

    "max_iterations": 100

And if you do not have any idea of the number of iterations required for the model to converge, keep it to the default
value and use the early stopper!

Early stopper
-------------

Pennylane does no have an early stopper implemented natively: we have implemented a simple one that allows to stop the calculation
when the validation loss does not improve or starts to increase due to overfitting. Activate it like this:

.. code-block:: python

    "use_early_stopper": True

Also, if you wish to change the patience of the early stopper - which is 20 iterations by default - use the keyword:

.. code-block:: python

    "early_stopper_patience": 50

Now, you have all the tools to generate a model that suits your needs. But this is not over! in order to get the most of
prevision-quantum-nn, you can use the preprocessing and postprocessing tools associated with the model! Let's start by
describing the preprocessing tools.

Tuning your preprocessing tools
===============================
        
.. code-block:: python

        preprocessing_params = {
            # preprocessing parameters
            # ...
        }
        qnn.get_application("classification", preprocessing_params=preprocessing_params)

Scaling
-------
If you use the angle encoding method, your features will need to be encoded as rotations. In order for the encoding to
be perfomed properly, we rescale all of your data to the range [0, :math:`\pi`]. This is done automatically.

Padding
-------
The first thing that you might have catched in the model section is:
what happens when my number of features is lower than the encoding capabilities? For example, what happens when the
number of qubits is 4, the encoding is angle and my number of features is only 2? Then, you need to resort to padding.
The default padding parameter is 0. This means that additional features will be constructed with the value 0. This is
the default behavior of applications. If you wish to change the padding parameter, you can use:

.. code-block:: python

        "padding": 0.1

.. warning::

        This is not recommended to change the default padding parameter for angle encoding as the features are scaled
        from 0 to pi. This feature is mainly used for amplitude encoding.

Feature construction
--------------------
Apart from padding and scaling preprocessing, there are ones that could improve the perfomance of your model! For
example, polynomial expansion. You can expand your features by applying a polynomial expansion of degree defined by:

.. code-block:: python

        "polynomial_degree": 2,

The default polynomial expansion method is ``"polynomial_features"``, provided by scikit-learn. You can change it to
``"kronecker"`` for a different type, but the first one works properly already!

.. code-block:: python

        "polynomial_expansion_type" : "polynomial_features" (default)

Feature engineering
-------------------
If you want something more sophisticated, use prevision.io's features enginnering.

.. code-block:: python

    "feature_engineering": True

.. warning::

        Prevision.io's feature engineering not accessible if you do not have prevision's library installed.

Dimension reduction
-------------------

This is very likely that your dataset already contains lots of features, and that you have already preprocessed it.
If your number of features exceeds the encoding capability of your hardware, use dimension reduction by setting the
following keywork to ``True``:

.. code-block:: python

        "force_dimension_reduction": True,

This is the default behavior of applications. If you want to encode a dataset that does not fit into the hardware
encoding capabilities, the library will run a dimension reduction method in order to lower the number of features.
A naive method would be to use a principal component analysis. We provide with such an option, but recommend to use a
wrapper instead. The wrapper will fit a LightGBM model on your dataset, then retain the best features according to the
importances computed during the training phase. Only the best features will be retained and provided to the encoding
method.

.. code-block:: python

        "dimension_reduction_fitter": "wrapper"
or

.. code-block:: python

        "dimension_reduction_fitter": "pca"

.. tip:: 

        Prefer the wrapper fitter compared to the PCA, it should work better!


Tuning your postprocesssing tools
=================================

Preprocessing was a necessary step, but what about postprocessing? There are situations where it is easy to visualize
the result of a model convergence.
In this section, we provide with tools to generate callbacks during the training phase so that you can monitor the decision boundaries of your classifier.
Define the postprocessing parameters and feed them to the application:

.. code-block:: python

        postprocessing_params = {
            # postprocessing params
            # ...
        }

        qnn.get_application("classification", postprocessing_params=postprocessing_params)

For now, only phase space plotting callbacks are implemented. We will implement more features in the future to allow you
to understand the quality of your decision boundaries!
For now, let's see how to generate plots as callbacks:
Define the plotting parameters as follows:

.. code-block:: python

        postprocessing_params = {
            "plotting_params": {
                "dim": 2, 
                "min_max_array": [[0, np.pi], [0, np.pi]],
                "verbose_period": 10,
                "prefix": "moon"
            }
        }
The parameter ``"dim"`` refers to the number of features in your dataset. Plotting is only allowed when the number of
features is 1 or 2. In this example, it is set to 2.

Prepare the plot limits by providing with the min and max of along each feature.

Then, use ``"val_verbose_period"`` to set the frequency (in number of iterations) at which the callback needs to be
called.

The resulting plots will have the name that you define: ``prefix_{iteration_number}.png``.

And you're good to go! Additional utilies are provided so that you can save your models, and reload them later.

Save an application
===================

Applications are automatically saved by calling the following two methods:

.. code-block:: python

        application.save_params()
        application.save_preprocessor()

A ``{prefix}_params.json`` and a ``{prefix}_preprocessor.obj``  files will be created in your current folder. 
You will then be able to load the application from the parameters json file. But this is only the structure of 
preprocessing, model and postprocessing objects. The preprocessor is loaded with the obj file.

In order to save the weights of the model during the training phase, you can set the keywork ``"snapshot_frequency"`` 
in the model_params:

.. code-block:: python

        "snapshot_frequency": 10

The weights of your model will be saved in ``"{prefix}_weights_{iteration_number}.npz"``.

Load an application
===================

You can load back an application to get into production for example:

.. code-block:: python

        application = qnn.load_application(application_params="moon_params.json",
                                           weights_file="moon_weights_100.npz"
                                           preprocessor_file="moon_preprocessor.obj")

Get into production!
====================
Finally, once your problem has been solved, get into production by calling:

.. code-block:: python

        predictions = application.predict(X_new)

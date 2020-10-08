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

Generating a Model
==================

.. code-block:: python

        import prevision_qnn as qnn

        model_params = {
            "architecture": "qubit",
            "num_q": 4,
            "num_categories": 2,
            "max_iterations": 1,
            "use_early_stopper": True,
            "save": True,
            "snapshot_frequency": 10,
            "prefix": "open_source",
            "num_layers": 5,
            "layer_type": "template",
            "optimizer_name": "Adam",
            "learning_rate": 0.05,
            "TYPE_problem": "classification",
            "batch_size": 5,
            "interface": "tf",
            "encoding": "angle",
        }
        model = qnn.get_model(model_params)
        model.build()

.. danger:: 
        The number of layers might be a tricky parameter to tune. For performance considerations, try not to increase it too much, or you will certainly
        fall into the well identified Barren plateau problem.

.. tip::
        From benchmarks results, it seems that the amplitude encoding perfoms worse compared to angle encoding. Try
        angle encoding first, even if you need to reduce the dimension of your features, it might be a better option
        than encoding thousands of features with amplitude encoding.

.. warning:: 
        Continuous Variable architectures simulations are heavy in memory. If you encounter a processed killed for this
        reason, try diminishing the "cutoff_dim" parameter of the model.

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

Set the number of qubits or qumodes used by the model with the ``num_q`` parameter.

.. code-block:: python

        "num_q": 10


Generating a Preprocessor
=========================
        
.. code-block:: python

        from prevision_qnn.preprocessing.preprocess import Preprocessor

        preprocessing_params = {
            "polynomial_degree": 2,
            "polynomial_expansion_type" : "polynomial_features",
            "force_dimension_reduction": True,
            "dimension_reduction_fitter": "wrapper",
            "padding": 0.3,
        }
        preprocessor = Preprocessor(preprocessing_params)
        preprocessor.build_for_model(model)

Generating a Postprocesssor
===========================

.. code-block:: python

        postprocessing_params = {
            "plotting_params": {
                "dim": 2, 
                "min_max_array": [[0, np.pi], [0, np.pi]],
                "verbose_period": 1,
                "prefix": "open_source"
            }
        }
        postprocessor = Preprocessor(postprocessing_params)

Save a model
============

.. code-block:: python

        "save": True

Fix the period in number of iterations you wish the model weights to be snapshoted.

.. code-block:: python

        "snapshot_frequency": 10

Load a model
============

.. code-block:: python

        import prevision_qnn as qnn
        model = qnn.get_model_from_parameters_file("params_file_name")
        model.load_weights("weight_file_name")

General input parameters
------------------------
It is intended to be lightweight and flexible.
The number of qubits is provided by the field ``"num_q"``.
The number of categories is provided by the field ``"num_categories"``,
Input the type of the problem so that the right metric is used during the optimization.
They can be of 4 different types:

1. ``"classification"``

2. ``"multiclassification"``

3. ``"regression"``

4. ``"reinforcement_learning"``

.. code-block:: python

   "num_q": 2,
   "num_categories": 2
   "TYPE_problem": "classification"

Encoding classical data into a quantum state
--------------------------------------------

The encoding mode depends on the architecture and on the library used for the variational circuit to be built on.
It is specified by the keyword ``encoding`` in the parameters' dictionary.

.. code-block:: python

    "encoding": "angle"

For qubit based architectures based on pennylane, there are three different types of encoding.

1. ``"angle"``

2. ``"amplitude"``

3. ``"mottonnen"``

For Continuous Variable based architectures based on pennylane, there are two different types of encoding.

1. ``"displacement"``

2. ``"squeezing"``

In the case of amplitude encoding, which encodes the data into 2^n amplitudes where n is the number of qubits, the
number of amplitudes might be greater than the number of features. But the entire vector of amplitudes needs to be
defined. We thus resort to padding. Change the padding parameter with the keyword ``padding``

.. code-block:: python

    "padding": 0.3

Building the layers of the circuit
----------------------------------

The layers of the neural network are automatically generated using the keywork ``num_layers``.
In this example, we set it to ``5``.
Also, use ``template`` if you wish to use pennylane's template type or ``custom`` if you wish to use the one defined by
Prevision.io. If the model is tensorflow-quantum, then this parameter will be set to ``custom`` anyways.

.. code-block:: python

    "num_layers": 5,
    "layer_type": "custom"

Early stopper
-------------
Pennylane has no early stopper implement natively: Prevision.io has developed one that allows to stop the calculation
when the validation loss does not improve or starts to increase due to overfitting. Activate it with the keyword
``use_early_stopper``.

.. code-block:: python

    "use_early_stopper": True

If the model is tensorflow-quantum, then it will use an early stopper based on ``keras``.

Classical feature engineering tricks
------------------------------------

Because inputing raw inputs to the model does not yield sufficiently efficient boundary decisions, it is common to
resort to some classical tricks.

The first trick is to use the polynomial expansion. Activate it by setting a ``polynomial_degree`` higher than 1.

.. code-block:: python

    "polynomial_degree": 2


If the polynomial expansion is not sufficient for your problem, use Prevision.io's feature engineering with the keyword
``feature_engineering``.

.. code-block:: python

    "feature_engineering": True


Dimension reduction
-------------------

Using the feature engineering will create additional features. In order to fit the number of features into the number of amplitudes or angles
available, use the keywork ``force_dimension_reduction`` to perform dimension reduction..

There are two types of dimension reduction possible.

1. ``"importance_sampling"`` which runs a LightGB model and retains the best features according to their importances.
   This is the default option.

2. ``"pca"``, which scaled down the number of features down to the available number of angles or amplitudes.

Manually set this option with the keyword ``dimension_reduction_fitter``.

.. code-block:: python

    "dimension_reduction_fitter": "importance_sampling"

The optimizer
-------------
You can set some parameters concerning the learning phase. For example, the optimizer name, the learning rate, or the
batch size.

.. code-block:: python

    "optimizer_name": "Adam",
    "learning_rate": 0.05,
    "batch_size": 5

The interface
-------------
When using pennylanes' models, you can choose the interface that will be used to compute the gradient. It can be of two
different types in prevision_qnn: ``tf``, which stands for tensorflow, or ``autograd``.

.. code-block:: python
    
    "interface": "tf"

The phase space plotter
-----------------------

In case you are running a 1D or a 2D problem, we have implemented a plotter that shows the current decision boundary.
Define it with the keywork ``plotting_params``. The dimension of the dataset needs to be input with ``"dim"``. Then ,the
min and max of the plot with ``min_max_array``. The plotter will execute and create a plot each ``verbose_period``. The
name of the file output will be prefixed by the keywork ``prefix``.

.. code-block:: python
    
    "plotting_params": {
        "dim": 2, 
        "min_max_array": np.array([[0, np.pi], [0, np.pi]]),
        "verbose_period": 10,
        "prefix": "test"
    }

The final input parameters
--------------------------

.. code-block:: python

        model_params = {
            "architecture": "qubit",
            "num_q": 4,
            "num_categories": 2,
            "max_iterations": 1,
            "use_early_stopper": True,
            "save": True,
            "snapshot_frequency": 10,
            "prefix": "open_source",
            "num_layers": 5,
            "layer_type": "template",
            "optimizer_name": "Adam",
            "learning_rate": 0.05,
            "TYPE_problem": "classification",
            "batch_size": 5,
            "interface": "tf",
            "encoding": "angle",
        }
        
        preprocessing_params = {
            "polynomial_degree": 2,
            "polynomial_expansion_type" : "polynomial_features",
            "feature_engineering": False,
            "padding": 0.3,
            "force_dimension_reduction": True,
            "dimension_reduction_fitter": "wrapper",
        }
        
        postprocessing_params = {
            "plotting_params": {
                "dim": 2, 
                "min_max_array": [[0, np.pi], [0, np.pi]],
                "verbose_period": 1,
                "prefix": "open_source"
            }
        }
        model = qnn.get_model(model_params)
        model.build()

Once the parameters of the model are defined, one can create the model importing the relevant modules of prevision-qnn.

First, you can import models that are built over ``pennylane``. The model that relies on qubit based architectures is
called ``PennylaneQubitNeuralNetwork``.

.. code-block:: python

    from prevision_qnn.qnn_pennylane import PennylaneQubitNeuralNetwork

In order to take advantage of photonics architectures as proposed by Xanadu, you can import ``CVNeuralNetwork``.

.. code-block:: python

    from prevision_qnn.qnn_pennylane import CVNeuralNetwork

If you wish to train a model using tensorflow quantum, import ``TensorflowNeuralNetwork`` and build the model with the
parameters' dictionary.

.. code-block:: python

    from prevision_qnn.qnn_tensorflow import TensorflowNeuralNetwork

Then, create the model depending on the architecture that you choose:

.. code-block:: python

    if architecture == "qubit"
        model = PennylaneQubitNeuralNetwork(params=params)
    elif architecture == "cv"
        model = CVNeuralNetwork(params=params)
    elif architecture == "tf"
        model = TensorflowNeuralNetwork(params=params)

Finally, simply call the method fit by inputing your training data and validation data. The validation data is not
mandatory, but if it is not provided, the early stopper will not be activated.

.. code-block:: python

    model.fit(X_train,
              y_train,
              val=X_test,
              val_labels=y_test,
              verbose=True)

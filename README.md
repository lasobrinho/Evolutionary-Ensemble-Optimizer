# Evolutionary Ensemble Optimizer

Optimizer for classifier ensembles based on Genetic Algorithm techniques. Given a trained ensemble of classifiers, the Evolutionary Ensemble Classifier is capable of finding best and more accurate subset of classifiers through evolutionary methods.

## Usage

The core file for the genetic optimization operation is the [optimizer.py](optimizers.py) file. 
There is an example on how to use the system using a Bagging Classifier with Decision Trees in the file [bagging_example.py](bagging_example.py). 
To change genetic algorithm parameters you need to open and modify [bagging_example.py](bagging_example.py). 
In order to run the system: 

```sh
$ python bagging_example.py
```
## Dependencies

The Evolutionary Ensemble Optimizer has the following dependencies:

* [Python 3]
* [Scikit Learn]
* [Colorama]

[//]: #
   [Python 3]: <https://www.python.org/download/releases/3.0/>
   [Scikit Learn]: <http://scikit-learn.org/>
   [Colorama]: <https://pypi.python.org/pypi/colorama>

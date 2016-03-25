Multi-Instance Learning for Sparsely Labelled Data
================================================

Sean Noran(<snoran@umass.edu>)

Overview
--------

In this work, we demonstrate that Multi-Instance Learning (MIL) can be applied to
various time-domain datasets, in order to reduce the need for fine-grained labels.
We examine various MIL algorithms. The Python implementations may be found at 
<https://github.com/garydoranjr/misvm>.

Installation
------------

This package can be installed in two ways (the easy way):

    # If needed:
    # pip install numpy
    # pip install scipy
    # pip install cvxopt
    pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm

or by running the setup file manually

    git clone [the url for misvm]
    cd misvm
    python setup.py install

Note the code depends on the `numpy`, `scipy`, and `cvxopt` packages. So have those
installed first. The build will likely fail if it can't find them. For more information, see:

 + [NumPy](http://www.numpy.org/): Library for efficient matrix math in Python
 + [SciPy](http://www.scipy.org/): Library for more MATLAB-like functionality
 + [CVXOPT](http://cvxopt.org/): Efficient convex (including quadratic program) optimization

Contents
--------

The MISVM package currently implements the following algorithms:

### SIL
Single-Instance Learning (SIL) is a "naive" approach that assigns each instance
the label of its bag, creating a supervised learning problem but mislabeling
negative instances in positive bags. It works surprisingly well for many
problems.
> Ray, Soumya, and Mark Craven. **Supervised versus multiple instance learning:
> an empirical comparison.** _Proceedings of the 22nd International Conference
> on Machine Learning._ 2005.

### MI-SVM and mi-SVM
These approaches modify the standard SVM formulation so that the constraints on
instance labels correspond to the MI assumption that at least one instance in
each bag is positive. For more information, see:
> Andrews, Stuart, Ioannis Tsochantaridis, and Thomas Hofmann. **Support vector
> machines for multiple-instance learning.** _Advances in Neural Information
> Processing Systems._ 2002.

### NSK and STK
The normalized set kernel (NSK) and statistics kernel (STK) approaches use
kernels to map entire bags into a features, then use the standard SVM
formulation to find bag classifiers:
> Gärtner, Thomas, Peter A. Flach, Adam Kowalczyk, and Alex J. Smola.
> **Multi-instance kernels.** _Proceedings of the 19th International Conference on
> Machine Learning._ 2002.

### MissSVM
MissSVM uses a semi-supervised learning approach, treating the instances in
positive bags as unlabeled data:
> Zhou, Zhi-Hua, and Jun-Ming Xu. **On the relation between multi-instance
> learning and semi-supervised learning.** _Proceedings of the 24th
> International Conference on Machine Learning._ 2007.

### MICA
The "multiple-instance classification algorithm" (MICA) represents each bag
using a convex combinations of its instances. The optimization program is then
solved by iteratively solving a series of linear programs. In our formulation,
we use L2 regularization, so we solve alternating linear and quadratic programs.
For more information on the original algorithm, see:
> Mangasarian, Olvi L., and Edward W. Wild. **Multiple instance classification
> via successive linear programming.** _Journal of Optimization Theory and
> Applications_ 137.3 (2008): 555-568.

### sMIL, stMIL, and sbMIL
This family of approaches intentionally bias SVM formulations to handle the
assumption that there are very few positive instances in each positive bag. In
the case of sbMIL, prior knowledge on the "sparsity" of positive bags can be
specified or found via cross-validation:
> Bunescu, Razvan C., and Raymond J. Mooney. **Multiple instance learning for
> sparse positive bags.** _Proceedings of the 24th International Conference on
> Machine Learning._ 2007.

How to Use
----------

The classifier implementations are loosely based on those found in the
[scikit-learn](http://scikit-learn.org/stable/) library. First, construct a
classifier with the desired parameters:

    >>> import misvm
    >>> classifier = misvm.MISVM(kernel='linear', C=1.0, max_iters=50)

Use Python's `help` functionality as in `help(misvm.MISVM)` or read the
documentation in the code to see which arguments each classifier takes. Then,
call the `fit` function with some data:

    >>> classifier.fit(bags, labels)

Here, the `bags` argument is a list of "array-like" (could be NumPy arrays, or a
list of lists) objects representing each bag. Each (array-like) bag has m rows
and f columns, which correspond to m instances, each with f features. Of course,
m can be different across bags, but f must be the same. Then `labels` is an
array-like object containing a label corresponding to each bag. *Each label must
be either +1 or -1.* You will likely get strange results if you try using
0/1-valued labels. After training the classifier, you can call the `predict`
function as:

    >>> labels = classifier.predict(bags)

Here `bags` has the same format as for `fit`, and the function returns an array
of real-valued predictions (use `numpy.sign(labels)` to get -1/+1 class
predictions).

An example script is included that trains classifiers on the [musk1
dataset](http://archive.ics.uci.edu/ml/datasets/Musk+(Version+1)); see:
> Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
> [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School
> of Information and Computer Science.

Install the package or add the `misvm` directory to the `PYTHONPATH` environment
variable before attempting to run the example using `python example.py` within
the `example` directory.

Questions and Issues
--------------------

If you find any bugs or have any questions about this code, please create an
issue on [GitHub](https://github.com/garydoranjr/misvm/issues), or contact Gary
Doran at <gary.doran@case.edu>. Of course, I cannot guarantee any support for
this software.

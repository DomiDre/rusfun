# RUSFUN

 [![Build Status](https://travis-ci.com/DomiDre/rusfun.svg?branch=master)](https://travis-ci.com/DomiDre/rusfun)
 [![](http://meritbadge.herokuapp.com/rusfun)](https://crates.io/crates/rusfun)
 
 The ``rusfun`` crate is a small library to compile parametrized functions from 
 Rust to wasm. Furthermore it contains minimizer routines to find for a given
 set of data, parameters that minimize a cost function.

 Currently the Levenberg-Marquardt algorithm is implemented to minimize 

 ![equation](https://latex.codecogs.com/svg.latex?%5Cchi%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cbiggl%28%20%5Cfrac%7By_i%20-%20f%28x_i%29%7D%7B%5Csigma_i%7D%20%5Cbiggr%29%5E2)
 

To define a function, a Func1D structs is defined, which contains as fields a
reference to the initial parameters p, a reference to the domain x and a function,
which maps p and x to the model values f(x).
A few models are pre-defined in the standard, size_distribution and sas modules.

To initiate a Gaussian function for example one can do:
```
let p = array![300.0, 3.0, 0.2, 0.0];
let model = size_distribution::gaussian;

let model_function = func1d::Func1D::new(&p, &x, model);
```
Note that p and x are [ndarrays](https://docs.rs/ndarray/0.12.1/ndarray/).

The function can then be evaluated by calling

```
model_function.output()
```


To minimize a model for given data (xᵢ, yᵢ, σᵢ) with LM a Minimizer struct needs 
to be initialized as mutable variable, with the previously defined model_function,
a reference to y and σ as ndarrays, as well as an initial ƛ value for the LM step.

```
let mut minimizer = curve_fit::Minimizer::init(&model_function, &y, &sy, 0.01);
```
Then a fit can be performed by
```
minimizer.fit()
```
and the result can be printed by
```
minimizer.report()
```


So far the basic function of the rusfun crate. The crate is very young and the
syntax might have breaking changes when more flexibility in choice for fitting
algorithms are implemented.

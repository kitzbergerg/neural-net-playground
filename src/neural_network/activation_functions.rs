#![allow(dead_code)]
use std::f32::consts::E;
use ndarray::Array2;

pub const SIGMOID: ApplyActivationFunction = ApplyActivationFunction {
    activation_function: |array| array.map(|x| sigmoid(*x)),
    derivative_of_activation_function: |array| array.map(|x| derivative_of_sigmoid(*x)),
};
pub const RELU: ApplyActivationFunction = ApplyActivationFunction {
    activation_function: |array| array.map(|x| re_lu(*x)),
    derivative_of_activation_function: |array| array.map(|x| derivative_of_re_lu(*x)),
};
pub const SOFTMAX: ApplyActivationFunction = ApplyActivationFunction {
    activation_function: |array| array.map(|x| *x / array.sum()),
    derivative_of_activation_function: |array| array, // there is no derivative
};


#[derive(Debug, Clone)]
pub struct ApplyActivationFunction {
    pub activation_function: fn(Array2<f32>) -> Array2<f32>,
    pub derivative_of_activation_function: fn(Array2<f32>) -> Array2<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivative_of_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn re_lu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

fn derivative_of_re_lu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { 1.0 }
}
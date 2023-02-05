use std::f32::consts::E;

pub fn activation_function(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x)) // sigmoid
    // if x < 0.0 { 0.0 } else { x } // ReLU
}

pub fn derivative_of_activation_function(x: f32) -> f32 {
    activation_function(x) * (1.0 - activation_function(x)) // sigmoid
    // if x < 0.0 { 0.0 } else { 1.0 } // ReLU
}
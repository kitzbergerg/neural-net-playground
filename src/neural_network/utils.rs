use std::f32::consts::E;

pub fn derivative_of_cost_function(x: f32) -> f32 {
    2.0 * x
}

pub fn activation_function(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x)) // sigmoid
    // if x < 0.0 { 0.0 } else { x } // ReLU
}

pub fn derivative_of_activation_function(x: f32) -> f32 {
    activation_function(x) * (1.0 - activation_function(x)) // sigmoid
    // if x < 0.0 { 0.0 } else { 1.0 } // ReLU
}

pub trait Transpose<T> {
    fn transpose(self) -> Vec<Vec<T>>;
}

impl<T> Transpose<T> for Vec<Vec<T>> {
    fn transpose(self) -> Vec<Vec<T>> {
        assert!(!self.is_empty());
        let len = self[0].len();
        let mut iters: Vec<_> = self.into_iter().map(|n| n.into_iter()).collect();
        (0..len).map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<_>>()
        })
            .collect()
    }
}
use ndarray::Array2;
use rand::{Rng, thread_rng};
use crate::neural_network::activation_functions::ApplyActivationFunction;

#[derive(Debug, Clone)]
pub(in crate::neural_network) struct Layer {
    pub(in crate::neural_network) left: usize,
    pub(in crate::neural_network) right: usize,
    pub(in crate::neural_network) learning_rate: f32,
    /// matrix is always (right x 1)
    pub(in crate::neural_network) biases: Array2<f32>,
    /// matrix is always (right x left)
    pub(in crate::neural_network) weights: Array2<f32>,
    /// matrix is always (left x 1)
    pub(in crate::neural_network) input_during_feedforward: Array2<f32>,
    pub(in crate::neural_network) activation_function: ApplyActivationFunction,
}

impl Layer {
    pub(in crate::neural_network) fn init_random(left: usize, right: usize, learning_rate: f32, activation_function: ApplyActivationFunction) -> Self {
        let mut rng = thread_rng();
        let mut random_vec = |len| (0..len).map(|_| rng.gen_range(-0.5..0.5)).collect::<Vec<_>>();
        Layer {
            left,
            right,
            learning_rate,
            biases: Array2::from_shape_vec((right, 1), random_vec(right)).unwrap(),
            weights: Array2::from_shape_vec((right, left), random_vec(right * left)).unwrap(),
            input_during_feedforward: Array2::default((0, 0)), // will be set later
            activation_function,
        }
    }

    pub(in crate::neural_network) fn feedforward_propagation(&mut self, input: Array2<f32>) -> Array2<f32> {
        assert_eq!(input.shape()[0], self.left);
        assert_eq!(input.shape()[1], 1);

        let pre_activation_function = &self.weights.dot(&input) + &self.biases;
        let output = (self.activation_function.activation_function)(pre_activation_function);

        assert_eq!(output.shape()[0], self.right);
        assert_eq!(output.shape()[1], 1);

        self.input_during_feedforward = input;
        output
    }

    pub(in crate::neural_network) fn backpropagation(&mut self, delta: &Array2<f32>) -> Array2<f32> {
        assert_eq!(delta.shape()[0], self.right);
        assert_eq!(delta.shape()[1], 1);

        // calc new weights and biases
        let delta_weights = self.learning_rate * delta.dot(&self.input_during_feedforward.t());
        let delta_biases = self.learning_rate * delta.sum();

        // calc next delta
        let pre_activation_function = self.weights.t().dot(delta);
        let new_delta = (self.activation_function.derivative_of_activation_function)(pre_activation_function);

        self.weights = &self.weights - delta_weights;
        self.biases = &self.biases - delta_biases;

        assert_eq!(new_delta.shape()[0], self.left);
        assert_eq!(new_delta.shape()[1], 1);

        new_delta
    }
}
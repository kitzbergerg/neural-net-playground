use crate::neural_network::activation_functions::{ApplyActivationFunction, SIGMOID};
use ndarray::{Array2, Axis};
use rand::{thread_rng, Rng};

#[derive(Debug, Clone)]
pub(in crate::neural_network) struct Layer {
    pub(in crate::neural_network) left: usize,
    pub(in crate::neural_network) right: usize,
    pub(in crate::neural_network) learning_rate: f32,
    /// matrix is always (right x left)
    pub(in crate::neural_network) weights: Array2<f32>,
    /// matrix is always (right x 1)
    pub(in crate::neural_network) biases: Array2<f32>,
    pub(in crate::neural_network) prev_layer_pre_activation_function: Option<Array2<f32>>,
    /// matrix is always (left x 1)
    pub(in crate::neural_network) prev_layer_output: Array2<f32>,
    pub(in crate::neural_network) activation_function: ApplyActivationFunction,
}

impl Layer {
    pub(in crate::neural_network) fn init_random(
        left: usize,
        right: usize,
        learning_rate: f32,
        activation_function: ApplyActivationFunction,
    ) -> Self {
        let mut rng = thread_rng();
        let mut random_vec = |len| {
            (0..len)
                .map(|_| rng.gen_range(-0.5..0.5))
                .collect::<Vec<_>>()
        };
        Layer {
            left,
            right,
            learning_rate,
            weights: Array2::from_shape_vec((right, left), random_vec(right * left)).unwrap(),
            biases: Array2::from_shape_vec((right, 1), random_vec(right)).unwrap(),
            prev_layer_pre_activation_function: Option::None, // will be set later
            prev_layer_output: Array2::default((0, 0)),       // will be set later
            activation_function,
        }
    }

    /// Returns values before activation function as first and layer output as second param
    pub(in crate::neural_network) fn feedforward_propagation(
        &mut self,
        (prev_layer_pre_activation_function, prev_layer_output): (Option<Array2<f32>>, Array2<f32>),
    ) -> (Option<Array2<f32>>, Array2<f32>) {
        assert_eq!(prev_layer_output.shape()[0], self.left);
        assert_eq!(prev_layer_output.shape()[1], 1);

        self.prev_layer_output = prev_layer_output;
        self.prev_layer_pre_activation_function = prev_layer_pre_activation_function;

        let pre_activation_function = &self.weights.dot(&self.prev_layer_output) + &self.biases;
        let layer_output =
            (self.activation_function.activation_function)(pre_activation_function.clone());

        assert_eq!(layer_output.shape()[0], self.right);
        assert_eq!(layer_output.shape()[1], 1);
        (Some(pre_activation_function), layer_output)
    }

    pub(in crate::neural_network) fn backpropagation(
        &mut self,
        delta: &Array2<f32>,
    ) -> Array2<f32> {
        assert_eq!(delta.shape()[0], self.right);
        assert_eq!(delta.shape()[1], 1);

        // calc new weights and biases
        let delta_weights = delta.dot(&self.prev_layer_output.t());
        let delta_biases = delta.sum_axis(Axis(1)).insert_axis(Axis(1)); // insert_axis is same as keepDims in python

        self.weights = &self.weights - self.learning_rate * delta_weights;
        self.biases = &self.biases - self.learning_rate * delta_biases;

        // calc next delta
        let mut new_delta = self.weights.t().dot(delta);
        if let Some(a) = &self.prev_layer_pre_activation_function {
            new_delta = new_delta * (SIGMOID.derivative_of_activation_function)(a.clone())
        }

        assert_eq!(new_delta.shape()[0], self.left);
        assert_eq!(new_delta.shape()[1], 1);

        new_delta
    }
}

use ndarray::Array2;
use rand::{Rng, thread_rng};
use crate::neural_network::utils::{activation_function, derivative_of_activation_function};

#[derive(Debug, Clone)]
pub struct NeuralNet {
    layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn init(learning_rate: f32, layer_sizes: &[usize]) -> Self {
        let mut layer_sizes_shifted_forward_by_1 = layer_sizes.iter();
        layer_sizes_shifted_forward_by_1.next();

        NeuralNet {
            layers: layer_sizes.iter().zip(layer_sizes_shifted_forward_by_1)
                .map(|(left, right)| Layer::init_random(*left, *right, learning_rate))
                .collect(),
        }
    }

    pub fn feedforward_propagation(&mut self, input: Vec<f32>) -> Vec<f32> {
        let initial_input = Array2::from_shape_vec((input.len(), 1), input).unwrap();

        let output = self.layers.iter_mut()
            .fold(initial_input, |input, layer| layer.feedforward_propagation(input, activation_function));

        output.into_raw_vec()
    }

    pub fn backpropagation(&mut self, actual: Vec<f32>, target: Vec<f32>) {
        assert_eq!(actual.len(), target.len());
        let actual = Array2::from_shape_vec((actual.len(), 1), actual).unwrap();
        let target = Array2::from_shape_vec((target.len(), 1), target).unwrap();
        let initial_delta = &actual - &target;

        let _ = self.layers.iter_mut()
            .rev()
            .fold(initial_delta, |delta, layer| layer.backpropagation(&delta, derivative_of_activation_function));
    }
}

#[derive(Debug, Clone)]
struct Layer {
    left: usize,
    right: usize,
    learning_rate: f32,
    /// matrix is always (right x 1)
    biases: Array2<f32>,
    /// matrix is always (right x left)
    weights: Array2<f32>,
    /// matrix is always (left x 1)
    input: Array2<f32>,
}

impl Layer {
    fn init_random(left: usize, right: usize, learning_rate: f32) -> Self {
        let mut rng = thread_rng();
        let mut random_vec = |len| (0..len).map(|_| rng.gen_range(-0.5..0.5)).collect::<Vec<_>>();
        Layer {
            left,
            right,
            learning_rate,
            biases: Array2::from_shape_vec((right, 1), random_vec(right)).unwrap(),
            weights: Array2::from_shape_vec((right, left), random_vec(right * left)).unwrap(),
            input: Array2::default((0, 0)), // will be set later
        }
    }

    fn feedforward_propagation(&mut self, input: Array2<f32>, activation_function: fn(f32) -> f32) -> Array2<f32> {
        assert_eq!(input.shape()[0], self.left);
        assert_eq!(input.shape()[1], 1);

        let output = (&self.weights.dot(&input) + &self.biases).map(|a| activation_function(*a));

        assert_eq!(output.shape()[0], self.right);
        assert_eq!(output.shape()[1], 1);

        self.input = input;
        output
    }

    fn backpropagation(&mut self, delta: &Array2<f32>, derivative_of_activation_function: fn(f32) -> f32) -> Array2<f32> {
        assert_eq!(delta.shape()[0], self.right);
        assert_eq!(delta.shape()[1], 1);

        self.weights = &self.weights - self.learning_rate * delta.dot(&self.input.t());
        self.biases = &self.biases - self.learning_rate * delta.sum();

        let new_delta = self.weights.t().dot(delta).map(|x| derivative_of_activation_function(*x));
        assert_eq!(new_delta.shape()[0], self.left);
        assert_eq!(new_delta.shape()[1], 1);

        new_delta
    }
}

#[cfg(test)]
mod test {
    use ndarray::arr2;
    use rand::Rng;
    use crate::get_index_of_maximum;
    use super::*;

    #[test]
    fn test_learning_with_2_node_network() {
        let target = 0.5;
        let input = 1.5;
        let initial_weight = 0.8;

        // 1 input node
        // 1 output node
        // initial weight of 0.8 between nodes
        let mut network = NeuralNet {
            layers: vec![
                Layer {
                    left: 1,
                    right: 1,
                    learning_rate: 0.1,
                    biases: arr2(&[[0.0]]),
                    weights: arr2(&[[initial_weight]]),
                    input: Default::default(),
                }
            ]
        };

        let mut prev_output = f32::MAX;
        let mut prev_weight = network.layers[0].weights.clone();
        for i in 0..10 {
            println!("----- Iteration: {i} -----");
            let actual = network.feedforward_propagation(vec![input]);

            network.backpropagation(actual.clone(), vec![target]);

            println!("Network output: {}, Target: {}", actual[0], target);
            println!("Old weight: {}, New weight: {}", prev_weight, &network.layers[0].weights);
            println!();

            // check if the network learns (approaches target)
            assert!(prev_output > actual[0]);

            // prepare for next iteration
            prev_weight = network.layers[0].weights.clone();
            prev_output = actual[0];
        }
    }

    #[test]
    fn test_network_with_function() {
        let mut rng = rand::thread_rng();
        let function = |x: f32| x;
        let mut training_data_generator = (0..1).cycle().map(|_| {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            let mut target = vec![0.0; 2];
            if function(x) > y { target[0] = 1.0 } else { target[1] = 1.0 };
            (vec![x, y], target)
        });

        let mut network = NeuralNet::init(0.1, &[2, 10, 2]);

        for i in 0..100 {
            let (input, target) = training_data_generator.next().unwrap();
            println!("----- Iteration: {i} -----");
            let actual = network.feedforward_propagation(input.clone());
            network.backpropagation(actual.clone(), target.clone());

            println!("input: {:?}", input);
            println!("output: {:?} (={})", actual, get_index_of_maximum(&actual) == 0);
            println!("target: {:?} (={})", target, target[0] == 1.0);
            println!();
        }
    }
}
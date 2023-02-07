use ndarray::{arr1, arr2, Array1, Array2, Axis};
use rand::{Rng, thread_rng};
use crate::neural_network::utils::{activation_function, derivative_of_activation_function};

#[derive(Debug)]
pub struct NeuralNet {
    learning_rate: f32,
    biases_per_layer: Vec<Array2<f32>>,
    weights_per_layer: Vec<Array2<f32>>,
    outputs_per_layer: Vec<Array2<f32>>,
}

impl NeuralNet {
    pub fn init(learning_rate: f32, layer_sizes: &[usize]) -> Self {
        let mut rng = thread_rng();
        let mut random_vec = |len| (0..len).map(|_| rng.gen_range(-0.5..0.5)).collect::<Vec<_>>();

        let biases_per_layer = (1..layer_sizes.len())
            .map(|i| layer_sizes[i])
            .map(|layer_size| Array2::from_shape_vec((layer_size, 1), random_vec(layer_size)).unwrap())
            .collect::<Vec<_>>();

        let mut layer_sizes_shifted_forward_by_1 = layer_sizes.iter();
        layer_sizes_shifted_forward_by_1.next();
        let weights_per_layer = layer_sizes.iter()
            .zip(layer_sizes_shifted_forward_by_1)
            .map(|(n, m)| Array2::from_shape_vec((*m, *n), random_vec(m * n)).unwrap())
            .collect::<Vec<_>>();

        let network = NeuralNet {
            learning_rate,
            biases_per_layer,
            weights_per_layer,
            outputs_per_layer: vec![Array2::default((0, 0)); layer_sizes.len() - 1], // dummy values. the actual values will be set during feedforward propagation
        };
        network
    }

    pub fn feedforward_propagation(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut outputs = Array2::from_shape_vec((input.len(), 1), input).unwrap();

        for i in 0..self.weights_per_layer.len() {
            outputs = (&self.weights_per_layer[i].dot(&outputs) + &self.biases_per_layer[i]).map(|a| activation_function(*a));
            self.outputs_per_layer[i] = outputs.clone();
        }

        outputs.into_raw_vec()
    }

    pub fn backpropagation(&mut self, actual: Vec<f32>, target: Vec<f32>) {
        assert_eq!(actual.len(), target.len());
        let actual = Array2::from_shape_vec((actual.len(), 1), actual).unwrap();
        let target = Array2::from_shape_vec((target.len(), 1), target).unwrap();
        let mut delta = &actual - &target;

        for i in self.weights_per_layer.len() - 1..=0 {
            self.weights_per_layer[i] = &self.weights_per_layer[i] - self.learning_rate * &delta.dot(&self.outputs_per_layer[i].t());
            self.biases_per_layer[i] = &self.biases_per_layer[i] - self.learning_rate * &delta;
            if i > 0 { delta = self.weights_per_layer[i].t().dot(&delta).map(|x| derivative_of_activation_function(*x)); }
        }
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;
    use crate::get_index_of_maximum;
    use super::*;

    #[test]
    fn test_learning_with_2_node_network() {
        let target = 0.5;
        let learning_rate = 0.1;
        let input = 1.5;
        let initial_weight = 0.8;

        // 1 input node
        // 1 output node
        // initial weight of 0.8 between nodes
        let mut network = NeuralNet {
            learning_rate,
            biases_per_layer: vec![arr1(&[0.0])],
            weights_per_layer: vec![arr2(&[[initial_weight]])],
            outputs_per_layer: vec![Array1::default(0)],
        };

        let mut prev_output = f32::MAX;
        let mut prev_weight = network.weights_per_layer[0].clone();
        for i in 0..10 {
            println!("----- Iteration: {i} -----");
            let actual = network.feedforward_propagation(vec![input]);

            network.backpropagation(actual.clone(), vec![target]);

            println!("Network output: {}, Target: {}", actual[0], target);
            println!("Old weight: {}, New weight: {}", prev_weight, &network.weights_per_layer[0]);
            println!();

            // check if the network learns (approaches target)
            assert!(prev_output > actual[0]);

            // prepare for next iteration
            prev_weight = network.weights_per_layer[0].clone();
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
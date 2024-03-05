use crate::neural_network::activation_functions::{SIGMOID, SOFTMAX};
use crate::neural_network::layer::Layer;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug, Clone)]
pub struct NeuralNet {
    layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn init(learning_rate: f32, layer_sizes: &[usize]) -> Self {
        let mut layers = layer_sizes
            .iter()
            .zip(layer_sizes[1..layer_sizes.len() - 1].iter())
            .map(|(left, right)| Layer::init_random(*left, *right, learning_rate, SIGMOID))
            .collect::<Vec<_>>();

        layers.push(Layer::init_random(
            layer_sizes[layer_sizes.len() - 2],
            layer_sizes[layer_sizes.len() - 1],
            learning_rate,
            SOFTMAX,
        ));

        NeuralNet { layers }
    }

    pub fn train(&mut self, mut data: Vec<(Vec<f32>, Vec<f32>)>) {
        data.shuffle(&mut thread_rng());
        for (input, target) in data {
            assert_eq!(self.layers.first().unwrap().left, input.len());
            assert_eq!(self.layers.last().unwrap().right, target.len());
            let actual = self.feedforward_propagation(input);
            self.backpropagation(actual, target);
        }
    }

    pub fn test(&mut self, data: Vec<(Vec<f32>, Vec<f32>)>) {
        let total = data.len();
        let mut sum = 0.0;

        for (input, label) in data {
            let actual = self.feedforward_propagation(input);
            if get_index_of_maximum(&actual) == get_index_of_maximum(&label) {
                sum += 1.0;
            }
        }

        println!(
            "Guessed {}/{} correctly ({}%)",
            sum,
            total,
            100.0 * sum / total as f32
        );
    }

    pub fn feedforward_propagation(&mut self, input: Vec<f32>) -> Vec<f32> {
        let initial_input = Array2::from_shape_vec((input.len(), 1), input).unwrap();

        self.layers
            .iter_mut()
            .fold((Option::None, initial_input), |input, layer| {
                layer.feedforward_propagation(input)
            })
            .1
            .into_raw_vec()
    }

    pub fn backpropagation(&mut self, actual: Vec<f32>, target: Vec<f32>) {
        assert_eq!(actual.len(), target.len());
        let actual = Array2::from_shape_vec((actual.len(), 1), actual).unwrap();
        let target = Array2::from_shape_vec((target.len(), 1), target).unwrap();
        let initial_delta = &actual - &target;

        let _ = self
            .layers
            .iter_mut()
            .rev()
            .fold(initial_delta, |delta, layer| layer.backpropagation(&delta));
    }
}

pub fn get_index_of_maximum(output: &Vec<f32>) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::neural_network::activation_functions::SIGMOID;
    use crate::neural_network::layer::Layer;
    use ndarray::arr2;
    use rand::Rng;

    #[test]
    fn test_converge_to_value() {
        let target = 0.5;
        let input = 1.5;
        let initial_weight = 0.8;

        // 1 input node
        // 1 output node
        // initial weight of 0.8 between nodes
        let mut network = NeuralNet {
            layers: vec![Layer {
                left: 1,
                right: 1,
                learning_rate: 0.1,
                weights: arr2(&[[initial_weight]]),
                biases: arr2(&[[0.0]]),
                prev_layer_pre_activation_function: Default::default(),
                prev_layer_output: Default::default(),
                activation_function: SIGMOID,
            }],
        };

        let mut prev_output = f32::MAX;
        for i in 0..10 {
            println!("----- Iteration: {i} -----");
            let actual = network.feedforward_propagation(vec![input]);

            network.backpropagation(actual.clone(), vec![target]);

            println!("Network output: {}", actual[0]);
            println!("New weight: {}", &network.layers[0].weights);
            println!();

            // check if the network learns (approaches target)
            assert!(prev_output > actual[0]);

            // prepare for next iteration
            prev_output = actual[0];
        }
    }

    #[test]
    fn test_network_with_function() {
        let mut rng = thread_rng();
        let function = |x: f32, y: f32| x > y;
        let mut data_generator = (0..1).cycle().map(|_| {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            let mut target = vec![0.0; 2];
            if function(x, y) {
                target[0] = 1.0
            } else {
                target[1] = 1.0
            };
            (vec![x, y], target)
        });

        let mut network = NeuralNet::init(0.1, &[2, 10, 10, 2]);

        println!("Training...");
        for _ in 0..10000 {
            let (input, target) = data_generator.next().unwrap();
            let actual = network.feedforward_propagation(input);
            network.backpropagation(actual, target);
        }

        println!("Testing...");
        network.test(data_generator.take(1000).collect());
    }

    #[test]
    fn test_and_logical_operators() {
        let mut rng = thread_rng();
        let map = [
            (vec![0.0, 0.0], vec![0.0, 1.0]),
            (vec![0.0, 1.0], vec![0.0, 1.0]),
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![1.0, 1.0], vec![1.0, 0.0]),
        ];
        let mut data_generator = (0..1)
            .cycle()
            .map(|_| map.get(rng.gen_range(0..3)).unwrap().clone());

        let mut network = NeuralNet::init(0.1, &[2, 10, 2]);

        println!("Training...");
        for _ in 0..10000 {
            let (input, target) = data_generator.next().unwrap();
            let actual = network.feedforward_propagation(input);
            // TODO: for some reason weight values just explode
            println!("{:?}", actual);
            network.backpropagation(actual, target);
        }

        println!("Testing...");
        network.test(data_generator.take(1000).collect());
    }
}

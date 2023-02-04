use std::f32::consts::E;
use std::mem::replace;
use itertools::izip;
use rand::Rng;
use crate::data::Data;

#[derive(Debug)]
pub struct NeuralNet {
    learning_rate: f32,
    layers: Vec<Vec<Node>>,
    output_layer: Vec<Node>,
}

impl NeuralNet {
    pub fn init(learning_rate: f32, layer_sizes: &[usize]) -> Self {
        let network = NeuralNet {
            learning_rate,
            layers: (0..layer_sizes.len() - 1)
                .map(|pos| Self::random_nodes(layer_sizes[pos], layer_sizes[pos + 1]))
                .collect(),
            output_layer: (0..layer_sizes[layer_sizes.len() - 1])
                .map(|_| Node {
                    prev_input: 0.0,
                    prev_output: 0.0,
                    bias: 0.0,
                    output_weights: vec![1.0],
                })
                .collect(),
        };
        println!("{:#?}", network);
        network
    }

    pub fn train(&mut self, data: Data) {
        println!("Training data size: {}", data.0.len());
        for (training, label) in data.0 {
            let input = training.iter().map(|x| *x as f32).collect::<Vec<_>>();
            let output = self.feedforward_propagation(input);

            let mut actual = vec![0.0; self.output_layer.len()];
            let _ = replace(&mut actual[label as usize], 1.0);
            let new_weights = self.backpropagation(output, actual);
            self.update_weights(new_weights);
        }
    }

    pub fn test(&mut self, data: Data) {
        println!("Test data size: {}", data.0.len());
        let total = data.0.len();
        let mut sum = 0.0;

        for (training, label) in data.0 {
            let input = training.iter().map(|x| *x as f32).collect::<Vec<_>>();
            let output = self.feedforward_propagation(input);
            println!("len: {}", output.len());
            let index_of_max = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            println!("Guess: {}, Label: {}", index_of_max, label);
            if index_of_max == label as usize { sum += 1.0; }
        }

        println!("{:#?}", self);
        println!("Guessed {}/{} correctly", sum, total);
        println!("Success rate: {}%", 100.0 * sum / total as f32);
    }

    pub fn feedforward_propagation(&mut self, input: Vec<f32>) -> Vec<f32> {
        let initial_input = input.into_iter().map(|i| vec![Edge { value: i, weight: 1.0 }]).collect::<Vec<_>>();

        // walk through all hidden layers
        let outputs = (0..self.layers.len() + 1).into_iter()
            .fold(initial_input, |inputs, current_layer| {
                if current_layer == self.layers.len() {
                    Self::calc_inputs_of_next_layer(&mut self.output_layer, inputs)
                } else {
                    Self::calc_inputs_of_next_layer(&mut self.layers[current_layer], inputs)
                }
            });

        // inner vec has only one value e.g. [[2],[3],...]
        let outputs = outputs.iter()
            .flatten()
            .map(|edge| edge.value)
            .collect::<Vec<_>>();

        // Calculate percentages
        let total = outputs.iter().sum::<f32>();
        outputs.iter().map(|output| *output / total).collect()
    }

    // TODO: that's prob wrong
    pub fn backpropagation(&self, actual: Vec<f32>, target: Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        // init first_term
        let init_first_term = actual.iter()
            .zip(target)
            .map(|(a, t)| derivative_of_cost_function(a - t))
            .collect::<Vec<_>>();

        let mut new_weights = Vec::new();

        vec![self.output_layer.clone()].iter()
            .chain(self.layers.iter().rev())
            .zip(self.layers.iter().rev())
            .fold(init_first_term, |first_term, (current_layer, prev_layer)| {
                // inputs_current_layer
                let second_term = current_layer.iter()
                    .map(|node| derivative_of_activation_function(node.prev_input))
                    .collect::<Vec<_>>();

                // outputs_prev_layer
                let third_term = prev_layer.iter()
                    .map(|node| node.prev_output)
                    .collect::<Vec<_>>();

                // calculate new weights
                let weights_from_prev_layer = prev_layer.iter()
                    .map(|node| node.output_weights.clone())
                    .collect::<Vec<_>>()
                    .transpose();
                let new_weights_for_layer = current_layer.iter().enumerate().map(|(i_current_layer, node)| {
                    weights_from_prev_layer[i_current_layer].iter().enumerate().map(|(i_prev_layer, w)| {
                        w - self.learning_rate * first_term[i_current_layer] * second_term[i_current_layer] * third_term[i_prev_layer]
                    }).collect::<Vec<_>>()
                })
                    .collect::<Vec<_>>()
                    .transpose();
                new_weights.push(new_weights_for_layer);

                // calculate first_term for next iteration
                prev_layer.iter().map(|node| {
                    izip!(&first_term, &second_term, &node.output_weights).map(|(f, s, w)| f * s * w).sum()
                })
                    .collect()
            });

        new_weights.reverse();
        new_weights
    }

    pub fn update_weights(&mut self, new_weights: Vec<Vec<Vec<f32>>>) {
        (0..self.layers.len())
            .into_iter()
            .zip(new_weights)
            .for_each(|(i, new_weights_for_layer)| {
                self.layers[i].iter_mut()
                    .zip(new_weights_for_layer)
                    .for_each(|(node, weights)| node.output_weights = weights)
            });
    }


    fn calc_inputs_of_next_layer(nodes: &mut Vec<Node>, inputs: Vec<Vec<Edge>>) -> Vec<Vec<Edge>> {
        nodes.iter_mut()
            .zip(inputs)
            .map(|(node, input)| {
                let output = node.calc_output(input);
                node.output_weights.iter().map(|weight| Edge { value: output, weight: *weight }).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .transpose()
    }

    fn random_nodes(size: usize, next_layer_size: usize) -> Vec<Node> {
        let mut vec = Vec::with_capacity(size);
        (0..size).for_each(|_| vec.push(Node::new_randomized(next_layer_size)));
        vec
    }
}

#[derive(Debug)]
struct Edge {
    value: f32,
    weight: f32,
}

#[derive(Debug, Clone)]
struct Node {
    prev_input: f32,
    prev_output: f32,
    bias: f32,
    output_weights: Vec<f32>,
}

impl Node {
    fn new_randomized(next_layer_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        Node {
            prev_input: 0.0,
            prev_output: 0.0,
            bias: rng.gen_range(0.0..1.0),
            output_weights: (0..next_layer_size).map(|_| rng.gen_range(0.0..1.0)).collect(),
        }
    }

    fn calc_output(&mut self, inputs: Vec<Edge>) -> f32 {
        let sum_of_input_times_weight = inputs.iter().map(|edge| edge.value * edge.weight).sum::<f32>();
        self.prev_input = sum_of_input_times_weight;
        let sum_with_bias = sum_of_input_times_weight + self.bias;
        let out = activation_function(sum_with_bias);
        self.prev_output = out;
        out
    }
}

fn derivative_of_cost_function(x: f32) -> f32 {
    2.0 * x
}

fn activation_function(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x)) // sigmoid
    // if x < 0.0 { 0.0 } else { x } // ReLU
}

fn derivative_of_activation_function(x: f32) -> f32 {
    activation_function(x) * (1.0 - activation_function(x)) // sigmoid
    // if x < 0.0 { 0.0 } else { 1.0 } // ReLU
}

trait Transpose<T> {
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

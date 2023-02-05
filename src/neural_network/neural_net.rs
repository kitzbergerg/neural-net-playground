use itertools::izip;
use crate::neural_network::edge::Edge;
use crate::neural_network::node::Node;
use crate::neural_network::utils::{derivative_of_activation_function, derivative_of_cost_function, Transpose};

#[derive(Debug)]
pub struct NeuralNet {
    learning_rate: f32,
    layers: Vec<Vec<Node>>,
}

impl NeuralNet {
    pub fn init(learning_rate: f32, layer_sizes: &[usize]) -> Self {
        let mut layers = (0..layer_sizes.len() - 1)
            .map(|pos| Self::random_nodes(layer_sizes[pos], layer_sizes[pos + 1]))
            .collect::<Vec<_>>();

        let output_layer = (0..layer_sizes[layer_sizes.len() - 1])
            .map(|_| Node {
                prev_input: 0.0,
                prev_output: 0.0,
                bias: 0.0,
                output_weights: vec![1.0],
            })
            .collect::<Vec<_>>();

        layers.push(output_layer);
        let network = NeuralNet {
            learning_rate,
            layers,
        };
        network
    }

    pub fn feedforward_propagation(&mut self, input: Vec<f32>) -> Vec<f32> {
        let initial_input = input.into_iter().map(|i| vec![Edge { value: i, weight: 1.0 }]).collect::<Vec<_>>();

        // walk through all hidden layers
        let outputs = self.layers.iter_mut()
            .fold(initial_input, |inputs, current_layer| {
                let next = Self::calc_inputs_of_next_layer(current_layer, inputs);
                next
            },
            );

        // inner vec has only one value e.g. [[2],[3],...]
        let outputs = outputs.iter()
            .flatten()
            .map(|edge| edge.value)
            .collect::<Vec<_>>();

        outputs
    }

    // TODO: that's prob wrong
    pub fn backpropagation(&self, actual: Vec<f32>, target: Vec<f32>) -> Vec<Vec<Vec<f32>>> {
        let init_first_term = actual.iter()
            .zip(target)
            .map(|(a, t)| derivative_of_cost_function(a - t))
            .collect::<Vec<_>>();

        let mut new_weights = Vec::new();

        let mut shifted_iter = self.layers.iter().rev();
        let _ = shifted_iter.next();

        self.layers.iter().rev()
            .zip(shifted_iter)
            .fold(init_first_term, |first_term, (current_layer, prev_layer)| {
                let second_term = current_layer.iter()
                    .map(|node| derivative_of_activation_function(node.prev_input))
                    .collect::<Vec<_>>();

                let third_term = current_layer.iter()
                    .enumerate()
                    .map(|(i, _node)| prev_layer.iter().map(|prev_node| prev_node.prev_output * prev_node.output_weights[i]).collect::<Vec<_>>())
                    .collect::<Vec<_>>();

                // calculate new weights
                let weights_from_prev_layer = prev_layer.iter()
                    .map(|node| node.output_weights.clone())
                    .collect::<Vec<_>>()
                    .transpose();
                let new_weights_for_layer = current_layer.iter()
                    .enumerate()
                    .map(|(i_of_node, _node)| {
                        weights_from_prev_layer[i_of_node].iter()
                            .enumerate()
                            .map(|(i_of_weight_of_node, w)| {
                                w - self.learning_rate * first_term[i_of_node] * second_term[i_of_node] * third_term[i_of_node][i_of_weight_of_node]
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

#[cfg(test)]
mod test {
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
            layers: vec![
                vec![Node {
                    prev_input: 0.0,
                    prev_output: 0.0,
                    bias: 0.0,
                    output_weights: vec![initial_weight],
                }],
                vec![Node {
                    prev_input: 0.0,
                    prev_output: 0.0,
                    bias: 0.0,
                    output_weights: vec![1.0],
                }],
            ],
        };

        let mut prev_output = f32::MAX;
        let mut prev_weight = network.layers[0][0].output_weights[0];
        for i in (0..100).into_iter() {
            println!("----- Iteration: {i} -----");
            let actual = network.feedforward_propagation(vec![input]);
            let new_weights = network.backpropagation(actual.clone(), vec![target]);
            network.update_weights(new_weights.clone());

            println!("Network output: {}, Target: {}", actual[0], target);
            println!("Old weight: {}, New weight: {}", prev_weight, new_weights[0][0][0]);
            println!();

            // check if weight were really updated
            assert_eq!(new_weights[0][0][0], network.layers[0][0].output_weights[0]);
            // check if the network learns (approaches target)
            assert!(prev_output > actual[0]);

            // prepare for next iteration
            prev_weight = new_weights[0][0][0];
            prev_output = actual[0];
        }
    }
}
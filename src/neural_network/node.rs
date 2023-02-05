use rand::Rng;
use crate::neural_network::edge::Edge;
use crate::neural_network::utils::activation_function;

#[derive(Debug, Clone)]
pub struct Node {
    pub(crate) prev_input: f32,
    pub(crate) prev_output: f32,
    pub(crate) bias: f32,
    pub(crate) output_weights: Vec<f32>,
}

impl Node {
    pub fn new_randomized(next_layer_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        Node {
            prev_input: 0.0,
            prev_output: 0.0,
            bias: rng.gen_range(0.0..1.0),
            output_weights: (0..next_layer_size).map(|_| rng.gen_range(0.0..1.0)).collect(),
        }
    }

    pub fn calc_output(&mut self, inputs: Vec<Edge>) -> f32 {
        let sum_of_input_times_weight = inputs.iter().map(|edge| edge.value * edge.weight).sum::<f32>();
        self.prev_input = sum_of_input_times_weight;
        let sum_with_bias = sum_of_input_times_weight + self.bias;
        let out = activation_function(sum_with_bias);
        self.prev_output = out;
        out
    }
}
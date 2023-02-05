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

    pub fn calc_output(&mut self, inputs: &Vec<Edge>) -> f32 {
        let input = inputs.iter().map(|edge| edge.value * edge.weight).sum::<f32>();
        self.prev_input = input;
        let output = activation_function(input + self.bias);
        self.prev_output = output;
        output
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_calc_output() {
        let mut node = Node {
            prev_input: 0.0,
            prev_output: 0.0,
            bias: 0.5,
            output_weights: vec![0.5],
        };
        let out = node.calc_output(&vec![
            Edge { value: 1.0, weight: 1.0 },
            Edge {
                value: 2.5,
                weight: 0.5,
            }]);

        assert_eq!(node.prev_input, 2.25);
        assert_eq!(out, activation_function(node.prev_input + node.bias));
    }
}
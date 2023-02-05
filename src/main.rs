use std::fs::File;
use std::mem::replace;
use std::path::Path;
use crate::data::Data;
use crate::neural_network::neural_net::NeuralNet;

mod data;
mod neural_network;

const NUMBER_OF_INPUTS: usize = 784;
const NUMBER_OF_OUTPUTS: usize = 10;
const LAYER_SIZES: [usize; 4] = [NUMBER_OF_INPUTS, 20, 10, NUMBER_OF_OUTPUTS];

fn main() {
    let data = Data::read_data_from_csv(File::open(Path::new("data/mnist_train.csv")).unwrap());
    let test_data = Data::read_data_from_csv(File::open(Path::new("data/mnist_test.csv")).unwrap());

    let mut neural_net = NeuralNet::init(0.1, &LAYER_SIZES);
    train(&mut neural_net, data, NUMBER_OF_OUTPUTS);
    test(&mut neural_net, test_data);
}


pub fn train(neural_network: &mut NeuralNet, data: Data, number_of_outputs: usize) {
    println!("Training data size: {}", data.0.len());
    for (training, label) in data.0 {
        let input = training.iter().map(|x| *x as f32).collect::<Vec<_>>();
        // print_images(label, &input);
        let output = neural_network.feedforward_propagation(input);
        // println!("output: {:?}", output);

        let mut actual = vec![0.0; number_of_outputs];
        let _ = replace(&mut actual[label as usize], 1.0);
        let new_weights = neural_network.backpropagation(output, actual);
        neural_network.update_weights(new_weights);
    }
}

pub fn test(neural_network: &mut NeuralNet, data: Data) {
    println!("Test data size: {}", data.0.len());
    let total = data.0.len();
    let mut sum = 0.0;

    for (training, label) in data.0 {
        let input = training.iter().map(|x| *x as f32).collect::<Vec<_>>();
        let output = neural_network.feedforward_propagation(input);
        println!("output: {:?}", output);
        let index_of_max = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        println!("Guess: {}, Label: {}", index_of_max, label);
        if index_of_max == label as usize { sum += 1.0; }
    }

    println!("{:#?}", neural_network);
    println!("Guessed {}/{} correctly", sum, total);
    println!("Success rate: {}%", 100.0 * sum / total as f32);
}

fn print_images(label: u8, input: &Vec<f32>) {
    println!("input:");
    for y in (0..28).into_iter() {
        for x in (0..28).into_iter() {
            print!("{}", if *input.get(y * 28 + x).unwrap() == 0.0 { " " } else { "X" });
        }
        println!()
    }
    println!("label: {}", label);
}
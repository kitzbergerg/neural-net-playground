use std::fs::File;
use std::path::Path;
use crate::data::Data;
use crate::neural_network::neural_net::NeuralNet;

mod data;
mod neural_network;

const NUMBER_OF_INPUTS: usize = 784;
const NUMBER_OF_OUTPUTS: usize = 10;
const LAYER_SIZES: [usize; 3] = [NUMBER_OF_INPUTS, 10, NUMBER_OF_OUTPUTS];

fn main() {
    let data = Data::read_data_from_csv(File::open(Path::new("data/mnist_train.csv")).unwrap());
    let test_data = Data::read_data_from_csv(File::open(Path::new("data/mnist_test.csv")).unwrap());

    let mut neural_net = NeuralNet::init(0.1, &LAYER_SIZES);
    for i in 0.. {
        println!("Epoch: {i}");
        neural_net.train(data.0.clone());
        neural_net.test(test_data.0.clone());
    }
}

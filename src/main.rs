use std::fs::File;
use std::path::Path;
use crate::data::Data;
use crate::neural_network::NeuralNet;

mod data;
mod neural_network;

fn main() {
    let number_of_inputs = 784;
    let number_of_outputs = 10;
    let mut neural_net = NeuralNet::init(0.1, &[number_of_inputs, 20, 10, number_of_outputs]);

    let data = Data::read_data_from_csv(File::open(Path::new("data/mnist_train.csv")).unwrap());

    neural_net.train(data);

    let test_data = Data::read_data_from_csv(File::open(Path::new("data/mnist_test.csv")).unwrap());
    neural_net.test(test_data);
}

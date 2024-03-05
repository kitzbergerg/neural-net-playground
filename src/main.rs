use crate::neural_network::neural_net::NeuralNet;

mod data;
mod neural_network;

const NUMBER_OF_INPUTS: usize = 784;
const NUMBER_OF_OUTPUTS: usize = 10;
const LAYER_SIZES: [usize; 3] = [NUMBER_OF_INPUTS, 30, NUMBER_OF_OUTPUTS];

#[tokio::main]
async fn main() {
    let (data, test_data) = data::get_data().await.unwrap();

    let mut neural_net = NeuralNet::init(0.1, &LAYER_SIZES);
    for i in 0..10 {
        println!("Epoch: {i}");
        neural_net.train(data.0.clone());
        neural_net.test(test_data.0.clone());
    }
}

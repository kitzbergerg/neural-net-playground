use std::{
    fs::File,
    io::{BufReader, Read, Write},
    mem::replace,
    path::Path,
};

use flate2::bufread::GzDecoder;

use crate::NUMBER_OF_OUTPUTS;

const DOWNLOAD_URL: &str = "http://yann.lecun.com/exdb/mnist/";
const FILENAME_TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
const FILENAME_TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
const FILENAME_TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
const FILENAME_TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

pub struct Data(pub Vec<(Vec<f32>, Vec<f32>)>);

pub async fn get_data() -> anyhow::Result<(Data, Data)> {
    // Download and process the train and test datasets
    tokio::try_join!(
        load_data(FILENAME_TRAIN_IMAGES, FILENAME_TRAIN_LABELS),
        load_data(FILENAME_TEST_IMAGES, FILENAME_TEST_LABELS)
    )
}

async fn load_data(filename_images: &str, filename_labels: &str) -> anyhow::Result<Data> {
    let (bytes_images, bytes_labels) =
        tokio::try_join!(download(filename_images), download(filename_labels))?;

    let data_images = parse_idx_images(&bytes_images)?;
    let data_labels = parse_idx_labels(&bytes_labels)?;

    let vec = data_images
        .into_iter()
        .zip(data_labels.into_iter())
        .collect();
    Ok(Data(vec))
}

async fn download(filename: &str) -> anyhow::Result<Vec<u8>> {
    let file_path = format!("data/{filename}");
    let file_path = Path::new(&file_path);

    if !file_path.exists() {
        println!("File {filename} does not exist. Downloading...");
        let response = reqwest::get(DOWNLOAD_URL.to_owned() + filename).await?;
        if !response.status().is_success() {
            return Err(response.error_for_status().unwrap_err().into());
        }

        let bytes = response.bytes().await?.to_vec();
        let mut f = File::create(file_path)?;
        f.write_all(&bytes)?;
    }

    let file = File::open(file_path)?;
    let file = BufReader::new(file);
    let mut file = GzDecoder::new(file);
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();

    return Ok(bytes);
}

fn parse_idx_images(bytes: &[u8]) -> anyhow::Result<Vec<Vec<f32>>> {
    let header_size = 16;
    let rows = 28;
    let cols = 28;
    let num_images = (bytes.len() - header_size) / (rows * cols);

    let data = bytes[header_size..]
        .iter()
        .map(|&pixel| pixel as f32 / 255.0)
        .collect::<Vec<f32>>();

    let vec = data
        .chunks_exact(rows * cols)
        .map(Vec::from)
        .collect::<Vec<_>>();

    assert_eq!(num_images, vec.len());

    Ok(vec)
}

fn parse_idx_labels(bytes: &[u8]) -> anyhow::Result<Vec<Vec<f32>>> {
    let header_size = 8;
    let labels = bytes[header_size..]
        .iter()
        .map(|&label| {
            let mut one_hot = vec![0.0; NUMBER_OF_OUTPUTS];
            if label < NUMBER_OF_OUTPUTS as u8 {
                let _ = replace(&mut one_hot[label as usize], 1.0);
            } else {
                println!("Error: Label should be in range 0..10, but is {label}");
            }
            one_hot
        })
        .collect();
    Ok(labels)
}

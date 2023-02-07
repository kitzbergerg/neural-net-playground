use std::fs::File;
use std::io::Read;
use std::mem::replace;
use std::str::FromStr;
use crate::NUMBER_OF_OUTPUTS;

pub struct Data(pub Vec<(Vec<f32>, Vec<f32>)>);


impl Data {
    pub fn read_data_from_csv(mut csv_file: File) -> Self {
        let mut csv_as_string = String::new();
        csv_file.read_to_string(&mut csv_as_string).unwrap();
        let csv_lines = csv_as_string.trim().split("\n").collect::<Vec<_>>();
        let data = csv_lines.into_iter()
            .map(|line| line.split(",").map(|number| u8::from_str(number).unwrap()).collect::<Vec<_>>())
            .map(|numbers_in_line| {
                let (a, b) = numbers_in_line.split_at(1);
                (b.to_vec(), a[0])
            })
            .map(|(inputs, label)| {
                let inputs = inputs.into_iter().map(|x| x as f32 / 256.0).collect::<Vec<_>>();
                let mut one_hot = vec![0.0; NUMBER_OF_OUTPUTS];
                let _ = replace(&mut one_hot[label as usize], 1.0);
                (inputs, one_hot)
            })
            .collect::<Vec<_>>();
        Data(data)
    }
}
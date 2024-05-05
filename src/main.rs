// use std::fs::File;
// use std::io::Write;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::prelude::*;
use ndarray::Array2;

fn main() {
    let feature_names = vec!["Slept Well", "Drank Coffee", "Ate Sushi", "Rust LoC"];
    let original_data: Array2<f32> = array!(
        [1., 1., 1., 900., 10.],
        [1., 1., 0., 500., 7.],
        [1., 0., 1., 200., 3.],
        [0., 1., 0., 300., 4.],
        [0., 0., 1., 100., 1.],
        [0., 0., 0., 500., 5.],
        [1., 1., 1., 0., 9.],
        [0., 0., 0., 0., 0.]
    );

    let num_features = original_data.len_of(Axis(1)) - 1;
    let features = original_data.slice(s![.., 0..num_features]).to_owned();
    let labels = original_data.column(num_features).to_owned();

    let linfa_dataset = Dataset::new(features, labels.clone())
        .map_targets(|x| match x.to_owned() as i32 {
            0..=3 => "Sad",
            4..=6 => "Ok",
            7..=10 => "Happy",
            _ => "Invalid Value",
        })
        .with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();
    let data = array!([1., 1., 1., 500.]).to_owned();
    let dataset = Dataset::new(data, labels);

    let result = model.predict(&dataset);
    println!("{}", result[0]);

    // File::create("dt.tex")
    //     .unwrap()
    //     .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
    //     .unwrap();
}

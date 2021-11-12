mod neuron;

use neuron::{Model, LinkingMethod::{EachToEach, OneToOne}};
use std::time::Instant;
use std::fs::File;
use std::io::Write;

fn main() {
  let now = Instant::now();
  
  let model = Model::builder()
    .neurons(256)
    .links(EachToEach, Some("ReLU"))
    .neurons(64)
    .links(EachToEach, Some("ReLU"))
    .neurons(64)
    .links(OneToOne, Some("Softmax"))
    .neurons(10)
    .build();

  println!("{:#?}", &model);

  let encoded: Vec<u8> = bincode::serialize(&model).unwrap();
  let mut file = File::create("foo.rnn").unwrap();
  file.write_all(&encoded).unwrap();

  let time = Instant::now().duration_since(now);

  // println!("Layer Chain: {:?}", layer_chain);
  println!("Time: {:?} ", time);
}
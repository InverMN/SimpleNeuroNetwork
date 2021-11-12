// VECTOR'S SLICES FOR REVERSED ORDER!!!
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Model {
  nodes: Vec<ModelNode>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuronsModelBuilder {
  nodes: Vec<ModelNode>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinksModelBuilder {
  nodes: Vec<ModelNode>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ModelNode {
  Neurons(Neurons),
  Links(Links),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Neurons {
  count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Links {
  method: LinkingMethod,
  activation: Option<String>,
  parameters: Option<Parameters>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LinkingMethod {
  EachToEach,
  OneToOne,
}

// all_inputs -> current_input -> output 

// parameters -> output -> all_errors -> current_error -> backprogated_error

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Parameters {
  biases: Vec<f64>,
  weights: Vec<f64>,
}

impl Model {
  
  pub fn builder() -> NeuronsModelBuilder {
    NeuronsModelBuilder {
      nodes: Vec::new(),
    }
  }

  fn predict(&self) {}

  fn backpropagate(&mut self) {}

  fn train(&mut self) {}

  fn import() -> () {}

  fn export(&self) {}
}

impl NeuronsModelBuilder {

  fn from_links_model_builder(links_model_builder: LinksModelBuilder) -> NeuronsModelBuilder {
    NeuronsModelBuilder {
      nodes: links_model_builder.nodes,
    }
  }

  pub fn neurons(mut self, count: usize) -> LinksModelBuilder {
    self.nodes.push(ModelNode::Neurons(Neurons { count }));
    LinksModelBuilder::from_neurons_model_builder(self)
  }
}

impl LinksModelBuilder {

  fn from_neurons_model_builder(neurons_model_builder: NeuronsModelBuilder) -> LinksModelBuilder {
    LinksModelBuilder {
      nodes: neurons_model_builder.nodes,
    }
  }

  pub fn links(mut self, method: LinkingMethod, activation: Option<&str>) -> NeuronsModelBuilder {
    self.nodes.push(
      ModelNode::Links(
        Links { 
          method,
          activation: activation.map(|it| it.to_owned()),
          parameters: None,
        }
      )
    );
    NeuronsModelBuilder::from_links_model_builder(self)
  }

  pub fn build(mut self) -> Model {
    let cloned = self.nodes.clone();

    self.nodes.iter_mut()
      .enumerate()
      .for_each(|(x, node)| {
        match node {
          ModelNode::Links(Links { method, parameters, .. }) => {
            let slice = cloned.as_slice();

            let input_size = match slice[x-1] {
              ModelNode::Neurons(Neurons { count }) => count,
              _ => 0,
            };

            let output_size = match slice[x+1] {
              ModelNode::Neurons(Neurons { count }) => count,
              _ => 0,
            };

            match method {
              LinkingMethod::EachToEach => {
                *parameters = Some(Parameters::new(output_size, input_size * output_size));
              },
              LinkingMethod::OneToOne => {
                *parameters = Some(Parameters::new(output_size, input_size));
              },
            }
          },
          _ => (),
        };
      });

    Model {
      nodes: self.nodes,
    }
  }
}

impl Parameters {
  fn new(biases_count: usize, weights_count: usize) -> Self {
    Parameters {
      biases: vec![0.1; biases_count],
      weights: vec![0.5; weights_count],
    }
  }
}
#include <torch/extension.h>

using torch::Tensor;

struct Container : torch::CustomClassHolder {
  Tensor x;
  Container() {}
  Container(Tensor x):x(x) {}
};

TORCH_LIBRARY(test, m) {
  m.class_<Container>("Container").def(
      torch::init<Tensor>());
}

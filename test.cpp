#include <torch/extension.h>
#include <iostream>

using torch::Tensor;
using torch::autograd::tensor_list;

struct Container : torch::CustomClassHolder {
  Tensor x;
  Container() {}
  Container(Tensor x):x(x) {}
  void release_resources() {
    std::cout << "--------ptr release-------- " <<"\n";
  }
};

class MyCustomFunc : public torch::autograd::Function<MyCustomFunc> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x) {
      at::AutoNonVariableTypeMode g;
      Container container = {torch::randn_like(x)};
      auto ptr = c10::make_intrusive<Container>(container);
      ctx->saved_data["x"] = ptr;
      Tensor t = container.x.clone().detach();
      return container.x;  // blow up, release_resources() got called, but ptr memory doesn't got free
      // return t;         // ptr memory got free
  }
  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grad_outputs) {
      return {grad_outputs[0] * MyCustomFunc::apply(ctx->saved_data["x"].toCustomClass<Container>()->x)};
  }
};

Tensor forward_autograd(Tensor x) {
   return MyCustomFunc::apply(x);
}

TORCH_LIBRARY(test1, m) {
  m.class_<Container>("Container").def(
      torch::init<Tensor>());
  m.def("forward", forward_autograd);
}

TORCH_LIBRARY_IMPL(test1, Autograd, m) {
  m.impl("forward", forward_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

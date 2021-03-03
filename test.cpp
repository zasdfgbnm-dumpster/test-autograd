#include <torch/extension.h>

using torch::Tensor;
using torch::autograd::tensor_list;

struct Container : torch::CustomClassHolder {
  Tensor x;
  Container() {}
  Container(Tensor x):x(x) {}
};

Tensor forward_cpu(Tensor x) {
    return torch::ones_like(x);
}

class MyCustomFunc : public torch::autograd::Function<MyCustomFunc> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx, Tensor x) {
      at::AutoNonVariableTypeMode g;
      ctx->saved_data["x"] = c10::make_intrusive<Container>(torch::randn_like(x));
      return forward_cpu(x);
  }
  static tensor_list backward(torch::autograd::AutogradContext* ctx, tensor_list grad_outputs) {
      at::AutoNonVariableTypeMode g;
      return {grad_outputs[0] * MyCustomFunc::apply(ctx->saved_data["x"].toCustomClass<Container>()->x)};
  }
};

Tensor forward_autograd(Tensor x) {
    MyCustomFunc::apply(x);
}

TORCH_LIBRARY(test, m) {
  m.class_<Container>("Container").def(
      torch::init<Tensor>());
  m.def("forward", forward_cpu);
}

TORCH_LIBRARY_IMPL(test, DefaultBackend, m) {
  m.impl("forward", forward_cpu);
}

TORCH_LIBRARY_IMPL(test, Autograd, m) {
  m.impl("forward", forward_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

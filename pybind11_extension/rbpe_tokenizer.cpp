#include "tokenizer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(rbpe_tokenizer, m) {
  m.doc() = "Type-safe RBPE Python bindings";

  py::class_<RBTokenizer>(m, "RBTokenizer")
      .def("save", &RBTokenizer::save, py::arg("path"))
      .def("load", &RBTokenizer::load, py::arg("path"))
      .def(py::init<int, const std::vector<std::string> &>(),
           py::arg("max_depth") = 0,
           py::arg("tech_terms") = std::vector<std::string>())
      .def("encode_with_dropout", &RBTokenizer::encode_with_dropout,
           py::arg("text"), py::arg("dropout_prob") = 0.1)
      .def("chunk_with_overlap", &RBTokenizer::chunk_with_overlap,
           py::arg("text"), py::arg("chunk_size") = 512,
           py::arg("overlap") = 64)
      .def(
          "encode",
          [](RBTokenizer &self, const std::string &text) {
            return self.encode(text);
          },
          py::arg("text"), "Encode text to token IDs")
      .def(
          "decode",
          [](RBTokenizer &self, const std::vector<int> &ids) {
            return self.decode(ids);
          },
          py::arg("ids"), "Decode token IDs to text")
      .def(
          "batch_encode",
          [](RBTokenizer &self, const std::vector<std::string> &texts) {
            std::vector<std::vector<int>> results;
            for (const auto &text : texts) {
              results.push_back(self.encode(text));
            }
            return results;
          },
          py::arg("texts"), "Batch encode texts")
      .def(
          "train",
          [](RBTokenizer &self, const std::string &corpus, int vocab_size) {
            if (corpus.empty())
              throw py::value_error("Corpus cannot be empty");
            if (vocab_size < 256)
              throw py::value_error("Vocab size must be ≥256");
            self.train(corpus, vocab_size);
          },
          py::arg("corpus"), py::arg("vocab_size"));
}

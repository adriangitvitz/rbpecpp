#include "tokenizer.h"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main() {
  RBTokenizer tokenizer(512);

  std::string inputFilePath = "data.txt";
  std::ifstream file(inputFilePath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << inputFilePath << std::endl;
    return 1;
  }

  std::string data((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  file.close();

  size_t n = data.size();
  std::string trainData = data.substr(0, static_cast<size_t>(n * 0.4));

  std::cout << "Training data" << std::endl;

  tokenizer.train(trainData, 30000);

  std::string text = "Before we proceed any further, hear me speak";

  std::cout << "Encoding text: " << text << std::endl;
  std::vector<int> encoded = tokenizer.encode(text);
  std::cout << "Encoded IDs: ";
  for (int id : encoded) {
    std::cout << id << " ";
  }
  std::cout << std::endl;

  std::string decoded = tokenizer.decode(encoded);
  std::cout << "Decoded: " << decoded << std::endl;

  return 0;
}

#ifndef INDEXED_LIST_H
#define INDEXED_LIST_H

#include "pair_hash.h"
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <utility>

class IndexedList {
public:
  struct Node {
    int val;
    Node *prev;
    Node *next;

    Node(int v) : val(v), prev(nullptr), next(nullptr) {}
  };

  IndexedList(const std::vector<uint8_t> &bytes) {
    if (bytes.empty()) {
      head = nullptr;
      tail = nullptr;
      size_ = 0;
      return;
    }
    head = new Node(bytes[0]);
    Node *current = head;
    for (size_t i = 1; i < bytes.size(); i++) {
      Node *newNode = new Node(bytes[i]);
      newNode->prev = current;
      current->next = newNode;
      current = newNode;

      add_to_index(current->prev->val, current->val, current->prev);
    }
    tail = current;
    size_ = bytes.size();
  }

  ~IndexedList() {
    Node *current = head;
    while (current) {
      Node *next = current->next;
      delete current;
      current = next;
    }
  }

  std::vector<Node *> &get_pair_positions(const std::pair<int, int> &pair) {
    return pair_index[pair];
  }

  void update_index(Node *node) {
    // Removing old pairs
    if (node->prev) {
      remove_from_index(node->prev->val, node->val, node->prev);
    }
    if (node->next) {
      remove_from_index(node->val, node->next->val, node);
    }

    // Adding new pairs
    if (node->prev) {
      add_to_index(node->prev->val, node->val, node->prev);
    }
    if (node->next) {
      add_to_index(node->val, node->next->val, node);
    }
  }

  Node *head = nullptr;
  Node *tail = nullptr;
  size_t size_ = 0;

private:
  std::unordered_map<std::pair<int, int>, std::vector<Node *>, pair_hash>
      pair_index;

  void add_to_index(int first, int second, Node *node) {
    auto key = std::make_pair(first, second);
    pair_index[key].push_back(node);
  }

  void remove_from_index(int first, int second, Node *node) {
    auto key = std::make_pair(first, second);
    auto &nodes = pair_index[key];
    // std::remove -> rearranges elements in [begin, end] so that all elements
    // not equal to node are moved to the front, returning an iterator
    auto new_end = std::remove(nodes.begin(), nodes.end(), node);
    nodes.erase(new_end, nodes.end());
    if (nodes.empty()) {
      pair_index.erase(key);
    }
  }
};

#endif // INDEXED_LIST_H

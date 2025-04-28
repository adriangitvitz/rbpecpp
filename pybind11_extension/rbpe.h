#ifndef Rbpe_H
#define Rbpe_H

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <deque>
#include <memory>
#include <stdio.h>
#include <string>
#include <unordered_map>

class CompressNode {
public:
  std::string prefix;
  std::unordered_map<std::string, std::shared_ptr<CompressNode>> children;
  int count;
  int value;
  time_t last_accessed;

  CompressNode(const std::string &prefix)
      : prefix(prefix), count(0), value(-1), last_accessed(std::time(nullptr)) {
  }

private:
  static double current_time() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }
};

class RadixBalancedTree {
public:
  RadixBalancedTree(size_t max_cache_size = 1024)
      : root(std::make_shared<CompressNode>("")),
        max_cache_size(max_cache_size) {
    for (int i = 0; i < 256; ++i) {
      byte_to_id[std::string(1, (char)i)] = i;
    }
  }

  int insert(const std::string &token_bytes, int token_id) {
    auto node = root;
    size_t i = 0;

    while (i < token_bytes.size()) {
      std::string byte = token_bytes.substr(i, 1);
      if (node->children.count(byte)) {
        auto child = node->children[byte];
        size_t prefix_len = child->prefix.size();

        if (token_bytes.substr(i, prefix_len) == child->prefix) {
          node = child;
          i += prefix_len;
        } else {
          size_t common_len = 0;
          size_t max_len = std::min(prefix_len, token_bytes.size() - 1);

          while (common_len < max_len &&
                 child->prefix[common_len] == token_bytes[i + common_len]) {
            common_len++;
          }

          auto split_node = std::make_shared<CompressNode>(
              child->prefix.substr(0, common_len));
          child->prefix = child->prefix.substr(common_len);
          split_node->children[child->prefix.substr(0, 1)] = child;

          std::string new_prefix = token_bytes.substr(i + common_len);
          auto new_node = std::make_shared<CompressNode>(new_prefix);
          new_node->value = token_id;
          split_node->children[new_prefix.substr(0, 1)] = new_node;
          node->children[byte] = split_node;
          node = new_node;
          i += common_len;
          break;
        }
      } else {
        auto new_node = std::make_shared<CompressNode>(token_bytes.substr(i));
        new_node->value = token_id;
        node->children[byte] = new_node;
        node = new_node;
        i = token_bytes.size();
        break;
      }
    }
    node->value = token_id;
    id_map[token_id] = token_bytes;
    return token_id;
  }

  int get_id(const std::string &token_bytes) {
    if (!root)
      return -1;

    auto current_node = root;
    size_t i = 0;

    while (i < token_bytes.size()) {
      std::string byte;
      try {
        byte = token_bytes.substr(i, 1);
      } catch (const std::exception &e) {
        return -1;
      }

      if (current_node->children.find(byte) == current_node->children.end()) {
        return -1;
      }

      auto child = current_node->children[byte];
      if (!child) {
        return -1;
      }

      const std::string &prefix = child->prefix;

      if (token_bytes.size() - i < prefix.size()) {
        return -1;
      }

      try {
        if (token_bytes.substr(i, prefix.size()) != prefix) {
          return -1;
        }
      } catch (const std::exception &e) {
        return -1;
      }

      i += prefix.size();
      current_node = child;
    }

    if (!current_node) {
      return -1;
    }

    if (current_node->value != -1) {
      try {
        current_node->last_accessed = current_time();
        update_cache(current_node);
      } catch (const std::exception &e) {
        // TODO: Handle error when updating cache
      }
    }

    return current_node->value;
  }

private:
  std::shared_ptr<CompressNode> root;
  std::deque<std::shared_ptr<CompressNode>> cache;
  std::unordered_map<int, std::string> id_map;
  std::unordered_map<std::string, int> byte_to_id;
  size_t max_cache_size;

  static double current_time() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  void update_cache(std::shared_ptr<CompressNode> node) {
    if (!node)
      return;

    auto it = std::find(cache.begin(), cache.end(), node);
    if (it != cache.end()) {
      cache.erase(it);
    }

    cache.push_front(node);

    if (cache.size() > max_cache_size) {
      cache.pop_back();
    }
  }

  std::shared_ptr<CompressNode>
  find_parent(std::shared_ptr<CompressNode> current,
              std::shared_ptr<CompressNode> parent,
              std::shared_ptr<CompressNode> target) {
    if (!current || !target)
      return nullptr;

    for (const auto &[key, child] : current->children) {
      if (!child)
        continue;

      if (child == target) {
        return parent;
      }
      auto result = find_parent(child, current, target);
      if (result != nullptr) {
        return result;
      }
    }
    return nullptr;
  }
};

#endif // !Rbpe_H

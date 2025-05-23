#ifndef Tokenizer_H
#define Tokenizer_H

#include "indexed_list.h"
#include "pairmultiset.h"
#include "rbpe.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <future>
#include <ios>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class RBTokenizer {
public:
  std::unordered_map<int, std::string> vocab;
  std::unordered_map<std::pair<int, int>, int, pair_hash> merges;
  std::unordered_set<std::string> tech_terms;
  std::mutex mtx;
  int max_depth;
  std::unique_ptr<RadixBalancedTree> rbt;

  RBTokenizer(int max_depth = 0,
              const std::vector<std::string> &tech_terms = {})
      : max_depth(max_depth), rbt(new RadixBalancedTree()) {
    for (int i = 0; i < 256; ++i) {
      vocab[i] = std::string(1, static_cast<char>(i));
    }
  }

  std::string decode(const std::vector<int> &ids) {
    std::string result;

    for (int id : ids) {
      if (vocab.find(id) != vocab.end()) {
        result += vocab[id];
      } else {
        result += static_cast<char>(id);
      }
    }
    return result;
  }

  std::vector<int> encode(const std::string &text) {
    std::vector<int> byte_stream(text.begin(), text.end());
    std::vector<int> ids;
    size_t pos = 0;

    CompressNode *current_node = rbt->root.get();
    int longest_match_id = -1;
    size_t longest_match_pos = 0;

    while (pos < byte_stream.size()) {
      uint8_t current_byte = byte_stream[pos];

      if (current_node->children.count(current_byte)) {
        auto child = current_node->children[current_byte];
        size_t prefix_len = child->prefix.size();

        if (text.compare(pos, prefix_len, child->prefix) == 0) {
          current_node = child.get();
          pos += prefix_len;

          if (current_node->value != -1) {
            longest_match_id = current_node->value;
            longest_match_pos = pos;
          }
          continue;
        }
      }

      if (longest_match_id != -1) {
        ids.push_back(longest_match_id);
        pos = longest_match_pos;
      } else {
        ids.push_back(static_cast<int>(byte_stream[pos]));
        pos++;
      }

      current_node = rbt->root.get();
      longest_match_id = -1;
      longest_match_pos = pos;
    }

    if (longest_match_id != -1) {
      ids.push_back(longest_match_id);
    }

    return ids;
  }

  void train(const std::string &text, int vocab_size,
             int merge_batch_size = 32) {
    std::vector<uint8_t> text_bytes(text.begin(), text.end());
    IndexedList list(text_bytes);
    PairMultiset stats;
    IndexedList::Node *curr = list.head;

    while (curr && curr->next) {
      stats.add({curr->val, curr->next->val});
      curr = curr->next;
    }
    int total_merges = vocab_size - 256 - merges.size();

    for (int i = 0; i < total_merges; i++) {
      auto [pair, count] = stats.max();
      if (count == 0) // No more pairs need to be merged
        break;

      // Creating new tokens
      int new_id = 256 + merges.size();
      std::string merged_bytes = vocab[pair.first] + vocab[pair.second];

      {
        std::lock_guard<std::mutex> lock(mtx);
        merges[pair] = new_id;
        vocab[new_id] = merged_bytes;
        rbt->insert(merged_bytes, new_id);
      }
      apply_merge(list, pair, new_id, stats);
    }
  }

  void apply_merge(IndexedList &list, const std::pair<int, int> &pair,
                   int new_id, PairMultiset &stats) {
    auto &positions = list.get_pair_positions(pair);

    for (auto node : positions) {
      if (!node || !node->next || node->val != pair.first ||
          node->next->val != pair.second) {
        continue;
      }

      if (node->prev) {
        stats.remove({node->prev->val, node->val});
      }

      if (node->next) {
        stats.remove({node->val, node->next->val});

        if (node->next->next) {
          stats.remove({node->next->val, node->next->next->val});
        }
      }

      IndexedList::Node *to_delete = node->next;
      IndexedList::Node *next_next = to_delete ? to_delete->next : nullptr;

      node->val = new_id;
      node->next = next_next;

      if (next_next) {
        next_next->prev = node;
      }

      delete to_delete;

      list.update_index(node);

      if (node->prev) {
        stats.add({node->prev->val, new_id});
      }

      if (node->next) {
        stats.add({new_id, node->next->val});
      }
    }
  }

  std::vector<int> encode_with_dropout(const std::string &text,
                                       float dropout_prob = 0.1) {
    std::vector<unsigned char> chars(text.begin(), text.end());
    std::vector<int> ids;

    size_t pos = 0;
    while (pos < chars.size()) {
      int best_token_id = -1;
      size_t best_length = 0;

      for (size_t len = 1;
           len <= std::min(static_cast<size_t>(max_depth), chars.size() - pos);
           len++) {
        std::string substr(chars.begin() + pos, chars.begin() + pos + len);
        int token_id = rbt->get_id(substr);

        if (token_id != -1) {
          bool should_apply =
              (len == 1) || ((double)rand() / RAND_MAX) > dropout_prob;

          if (should_apply && len > best_length) {
            best_token_id = token_id;
            best_length = len;
          }
        }
      }

      if (best_length > 0) {
        ids.push_back(best_token_id);
        pos += best_length;
      } else {
        ids.push_back(chars[pos]);
        pos++;
      }
    }

    return ids;
  }

  std::vector<std::vector<int>> chunk_with_overlap(const std::string &text,
                                                   int chunk_size = 512,
                                                   int overlap = 64) {
    std::vector<int> tokens = encode(text);
    std::vector<std::vector<int>> chunks;

    if (static_cast<int>(tokens.size()) <= chunk_size) {
      chunks.push_back(tokens);
      return chunks;
    }

    size_t start = 0;
    while (start < tokens.size()) {
      size_t end = std::min(start + chunk_size, tokens.size());
      std::vector<int> chunk(tokens.begin() + start, tokens.begin() + end);
      chunks.push_back(chunk);

      start += (chunk_size - overlap);
      if (start >= tokens.size())
        break;
    }

    return chunks;
  }

  void save(const std::string &path) {
    std::lock_guard<std::mutex> lock(mtx);
    std::ofstream out(path, std::ios::binary);

    size_t vocab_size = vocab.size();
    out.write(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));
    for (const auto &[id, bytes] : vocab) {
      out.write(reinterpret_cast<const char *>(&id), sizeof(id));
      size_t len = bytes.size();
      out.write(reinterpret_cast<const char *>(&len), sizeof(len));
      out.write(bytes.data(), len);
    }

    size_t merges_size = merges.size();
    out.write(reinterpret_cast<char *>(&merges_size), sizeof(merges_size));
    for (const auto &[pair, id] : merges) {
      out.write(reinterpret_cast<const char *>(&pair.first),
                sizeof(pair.first));
      out.write(reinterpret_cast<const char *>(&pair.second),
                sizeof(pair.second));
      out.write(reinterpret_cast<const char *>(&id), sizeof(id));
    }
    serialize_tree(out, rbt->root);
  }

  void load(const std::string &path) {
    std::lock_guard<std::mutex> lock(mtx);
    std::ifstream in(path, std::ios::binary);

    // Clear existing state
    vocab.clear();
    merges.clear();
    rbt.reset(new RadixBalancedTree());

    // Load vocab
    size_t vocab_size;
    in.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));
    for (size_t i = 0; i < vocab_size; ++i) {
      int id;
      size_t len;
      in.read(reinterpret_cast<char *>(&id), sizeof(id));
      in.read(reinterpret_cast<char *>(&len), sizeof(len));
      std::string bytes(len, '\0');
      in.read(&bytes[0], len);
      vocab[id] = bytes;
    }

    // Load merges
    size_t merges_size;
    in.read(reinterpret_cast<char *>(&merges_size), sizeof(merges_size));
    for (size_t i = 0; i < merges_size; ++i) {
      std::pair<int, int> pair;
      int id;
      in.read(reinterpret_cast<char *>(&pair.first), sizeof(pair.first));
      in.read(reinterpret_cast<char *>(&pair.second), sizeof(pair.second));
      in.read(reinterpret_cast<char *>(&id), sizeof(id));
      merges[pair] = id;
    }

    rebuild_tree(in, rbt->root);
  }

private:
  std::unordered_set<std::string>
  init_tech_term(const std::vector<std::string> &terms) {
    std::unordered_set<std::string> term_set;
    for (const auto &term : terms) {
      term_set.insert(term);
    }
    return term_set;
  }

  std::unordered_map<std::pair<int, int>, int, pair_hash>
  count_pairs(const std::vector<int> &seq) {
    std::unordered_map<std::pair<int, int>, int, pair_hash> pairs;
    for (size_t i = 0; i < seq.size() - 1; ++i) {
      std::pair<int, int> p(seq[i], seq[i + 1]);
      pairs[p]++;
    }
    return pairs;
  }

  void premerge_technical_terms(const std::string &text) {
    for (const auto &term : tech_terms) {
      if (rbt->get_id(term) == -1) {
        std::vector<int> current_seq(term.begin(), term.end());
        while (current_seq.size() > 1) {
          std::unordered_map<std::pair<int, int>, int, pair_hash> pairs;
          for (size_t i = 0; i < current_seq.size() - 1; ++i) {
            pairs[{current_seq[i], current_seq[i + 1]}]++;
          }

          if (pairs.empty())
            break;

          // best_pair = max(pairs, key=lambda k: (pairs[k], -k[0], -k[1]))
          auto best_pair_it = std::max_element(
              pairs.begin(), pairs.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });
          std::pair<int, int> best_pair = best_pair_it->first;
          // std::lock_guard<std::mutex> lock(mtx);
          int new_id = 256 + merges.size();
          std::string merged_bytes =
              vocab[best_pair.first] + vocab[best_pair.second];
          merges[best_pair] = new_id;
          vocab[new_id] = merged_bytes;
          rbt->insert(merged_bytes, new_id);
          current_seq = replace_pair(current_seq, best_pair, new_id);
        }
      }
    }
  }

  std::vector<int> replace_pair(const std::vector<int> &seq,
                                const std::pair<int, int> &pair_to_replace,
                                int new_id) {
    std::vector<int> new_seq;

    size_t i = 0;
    while (i < seq.size()) {
      if (i < seq.size() - 1 &&
          std::make_pair(seq[i], seq[i + 1]) == pair_to_replace) {
        new_seq.push_back(new_id);
        i += 2;
      } else {
        new_seq.push_back(seq[i]);
        i += 1;
      }
    }
    return new_seq;
  }

  std::vector<int> replace_sequence_with_replacements(
      const std::vector<int> &seq,
      const std::unordered_map<std::pair<int, int>, int, pair_hash>
          &replacements) {
    std::vector<int> new_seq;

    size_t i = 0;
    while (i < seq.size()) {
      bool replaced = false;

      if (i < seq.size() - 1 && replacements.count({seq[i], seq[i + 1]})) {
        new_seq.push_back(replacements.at({seq[i], seq[i + 1]}));
        i += 2;
        replaced = true;
      }

      if (!replaced) {
        new_seq.push_back(seq[i]);
        ++i;
      }
    }

    return new_seq;
  }

  void serialize_tree(std::ostream &out,
                      const std::shared_ptr<CompressNode> &node) {
    if (!node)
      return;

    // Serialize current node
    size_t prefix_len = node->prefix.size();
    out.write(reinterpret_cast<const char *>(&prefix_len), sizeof(prefix_len));
    out.write(node->prefix.data(), prefix_len);
    out.write(reinterpret_cast<const char *>(&node->value),
              sizeof(node->value));

    size_t num_children = node->children.size();
    out.write(reinterpret_cast<const char *>(&num_children),
              sizeof(num_children));
    for (const auto &[byte_key, child] : node->children) {
      out.write(reinterpret_cast<const char *>(&byte_key), sizeof(uint8_t));
      serialize_tree(out, child);
    }
  }

  void rebuild_tree(std::istream &in, std::shared_ptr<CompressNode> &node) {
    size_t prefix_len;
    in.read(reinterpret_cast<char *>(&prefix_len), sizeof(prefix_len));
    std::string prefix(prefix_len, '\0');
    in.read(&prefix[0], prefix_len);

    int value;
    in.read(reinterpret_cast<char *>(&value), sizeof(value));

    node = std::make_shared<CompressNode>(prefix);
    node->value = value;

    size_t num_children;
    in.read(reinterpret_cast<char *>(&num_children), sizeof(num_children));
    for (size_t i = 0; i < num_children; ++i) {
      uint8_t byte_key;
      in.read(reinterpret_cast<char *>(&byte_key), sizeof(uint8_t));
      std::shared_ptr<CompressNode> child;
      rebuild_tree(in, child);
      node->children[byte_key] = child;
    }
  }
};

#endif // !Tokenizer_H

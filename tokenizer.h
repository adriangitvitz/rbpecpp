#ifndef Tokenizer_H
#define Tokenizer_H

#include "rbpe.h"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct pair_hash {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2> &p) const {
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

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
    size_t i = 0;

    while (i < byte_stream.size()) {
      size_t max_len =
          std::min(static_cast<size_t>(max_depth), byte_stream.size() - 1);
      bool found = false;
      for (size_t length = max_len; length > 0; --length) {
        std::string chunk(byte_stream.begin() + i,
                          byte_stream.begin() + i + length);
        int token_id = rbt->get_id(chunk);
        if (token_id != -1) {
          ids.push_back(token_id);
          i += length;
          found = true;
          break;
        }
      }
      if (!found) {
        ids.push_back(byte_stream[i]);
        ++i;
      }
    }
    return ids;
  }

  void train(const std::string &text, int vocab_size,
             int merge_batch_size = 32) {
    std::vector<int> text_bytes(text.begin(), text.end());
    std::vector<std::vector<int>> ids = {text_bytes};

    premerge_technical_terms(text);

    int total_merges = vocab_size - 256 - merges.size();
    int iterations = std::max(1, total_merges / merge_batch_size);

    for (int iter = 0; iter < iterations; ++iter) {
      std::unordered_map<std::pair<int, int>, int, pair_hash> batch_stats;
      std::vector<
          std::future<std::unordered_map<std::pair<int, int>, int, pair_hash>>>
          futures;
      for (const auto &seq : ids) {
        futures.push_back(std::async(&RBTokenizer::count_pairs, this, seq));
      }

      for (auto &future : futures) {
        auto result = future.get();
        for (const auto &[pair, count] : result) {
          batch_stats[pair] += count;
        }
      }
      std::vector<std::pair<std::pair<int, int>, int>> top_pairs(
          batch_stats.begin(), batch_stats.end());
      std::sort(top_pairs.begin(), top_pairs.end(),
                [](auto &a, auto &b) { return a.second > b.second; });
      std::vector<std::pair<int, int>> current_merges;
      for (int i = 0;
           i < std::min(merge_batch_size, static_cast<int>(top_pairs.size()));
           ++i) {
        auto [pair, _] = top_pairs[i];
        current_merges.push_back(pair);
        int new_id = 256 + merges.size();
        std::string merged_bytes = vocab[pair.first] + vocab[pair.second];
        std::lock_guard<std::mutex> lock(mtx);
        merges[pair] = new_id;
        vocab[new_id] = merged_bytes;
        rbt->insert(merged_bytes, new_id);
      }
      ids = parallel_batch_replace(ids, current_merges);
    }
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
          std::lock_guard<std::mutex> lock(mtx);
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

  std::vector<std::vector<int>> parallel_batch_replace(
      const std::vector<std::vector<int>> &sequences,
      const std::vector<std::pair<int, int>> &current_merges) {
    std::unordered_map<std::pair<int, int>, int, pair_hash> replacements;
    for (const auto &merge : current_merges) {
      replacements[merge] = merges[merge];
    }

    std::vector<std::future<std::vector<int>>> futures;

    for (const auto &seq : sequences) {
      futures.push_back(
          std::async(&RBTokenizer::replace_sequence_with_replacements, this,
                     seq, replacements));
    }
    std::vector<std::vector<int>> results;
    for (auto &future : futures) {
      results.push_back(future.get());
    }
    return results;
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
};

#endif // !Tokenizer_H

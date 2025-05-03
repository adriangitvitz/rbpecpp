#ifndef PAIR_MULTISET_H
#define PAIR_MULTISET_H

#include "pair_hash.h"
#include <functional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

class PairMultiset {
public:
  void add(const std::pair<int, int> &pair, int count = 1) {
    counts_[pair] += count;
    heap_.emplace(pair, counts_[pair]);
  }

  void remove(const std::pair<int, int> &pair, int count = 1) {
    auto it = counts_.find(pair);
    if (it == counts_.end())
      return;

    it->second -= count;
    if (it->second <= 0) {
      counts_.erase(it);
    } else {
      heap_.emplace(pair, it->second);
    }
  }

  std::pair<std::pair<int, int>, int> max() {
    while (!heap_.empty()) {
      auto current = heap_.top();
      auto it = counts_.find(current.first);

      if (it != counts_.end() && it->second == current.second) {
        return current;
      }

      heap_.pop();
    }
    return {{-1, -1}, 0};
  }

  size_t size() const { return counts_.size(); }
  bool empty() const { return counts_.empty(); }

private:
  struct Compare {
    bool operator()(const std::pair<std::pair<int, int>, int> &a,
                    const std::pair<std::pair<int, int>, int> &b) {
      return a.second < b.second; // Max-heap
    }
  };

  std::priority_queue<std::pair<std::pair<int, int>, int>,
                      std::vector<std::pair<std::pair<int, int>, int>>, Compare>
      heap_;

  std::unordered_map<std::pair<int, int>, int, pair_hash> counts_;
};

#endif // PAIR_MULTISET_H

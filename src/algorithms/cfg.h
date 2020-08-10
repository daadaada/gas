#pragma once

#include <vector>

#include "src/core/kernel.h"

namespace dada {
  class BasicBlock;
  class Edge;
  class CFG;

  class BasicBlock {
    public:
      int start_;
      int end_;
      std::vector<Edge*> inEdges_;
      std::vector<Edge*> outEdges_;
      friend CFG;
    public:
      BasicBlock(int start);

      void setEnd(int end);

      Edge* addSuccessor(BasicBlock* successor);
      Edge* addPredecessor(BasicBlock* predecessor);
  };

  class Edge {
    public:
      BasicBlock* tail_;
      BasicBlock* head_;
      friend CFG;
    public:
      Edge(BasicBlock* tail, BasicBlock* head);
  };

  class CFG {
    public:
      std::vector<BasicBlock*> basic_blocks_;
      std::vector<Edge*> edges_;
      BasicBlock* root_;

    public:
      CFG(Kernel const* kernel);
      ~CFG();

      void printCFG() const;
  };
} // namespace dada
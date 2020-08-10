#include <map>
#include <algorithm> // std::find_if
#include <iostream>

#include "cfg.h"

using namespace std;

namespace dada {
//----------------------BasicBlock-------------------------
  BasicBlock::BasicBlock(int start) : start_(start){}

  void BasicBlock::setEnd(int end) { end_ = end; }

  Edge* BasicBlock::addSuccessor(BasicBlock* successor){
    Edge* edge = new Edge(this, successor);
    outEdges_.push_back(edge);

    successor->inEdges_.push_back(edge);

    return edge;
  }

  Edge* BasicBlock::addPredecessor(BasicBlock* predecessor){
    if(predecessor == nullptr) return nullptr;

    Edge* edge = new Edge(predecessor, this);
    inEdges_.push_back(edge);
    
    predecessor->outEdges_.push_back(edge);

    return edge;
  }

//-------------------------Edge----------------------------
  Edge::Edge(BasicBlock* tail, BasicBlock* head) : tail_(tail), head_(head){}

//--------------------------CFG----------------------------
  CFG::CFG(Kernel const* kernel){
    if(kernel->instructions_.size() == 0) return;

    map<int, BasicBlock*> visited_bbs;
    map<int, vector<BasicBlock*>> to_visit_bbs;

    to_visit_bbs[0] = vector<BasicBlock*>{};

    auto tryAddSuccessor = [&to_visit_bbs, &visited_bbs, this](BasicBlock* cbb, int idx){
      if(visited_bbs.find(idx) != visited_bbs.end()){
        BasicBlock* target_bb = visited_bbs.at(idx);
        edges_.push_back(cbb->addSuccessor(target_bb));
      } else if(to_visit_bbs.find(idx) != to_visit_bbs.end()){
        to_visit_bbs.at(idx).push_back(cbb);
      } else {
        to_visit_bbs.insert(make_pair(idx, vector<BasicBlock*>{cbb}));
      }
    };

    while(!to_visit_bbs.empty()){
      auto [start_idx, predecessors] = *(to_visit_bbs.begin());
      to_visit_bbs.erase(to_visit_bbs.begin());

      BasicBlock* cbb = new BasicBlock(start_idx);
      if(basic_blocks_.size() == 0) root_ = cbb;
      basic_blocks_.push_back(cbb);
      visited_bbs.insert(make_pair(start_idx, cbb));

      for(BasicBlock* predecessor : predecessors){
        if(predecessor != nullptr) edges_.push_back(cbb->addPredecessor(predecessor));
      }      



      int i=start_idx;
      for(; ; ++i){
        if(i >= kernel->instructions_.size()){
          throw runtime_error("Kernel must end with exit/bra.\n");
        }
        auto const& instr = kernel->instructions_[i];
        if(instr->opcode_ == EXIT){
          if(instr->predicate_mask_){ // conditional exit
            continue;
          } else { // unconditional branch
            cbb->setEnd(i);
            break;
          }
        } else if(instr->opcode_ == BRA){
          string label_name = instr->src_operands_[0]->name_;
          int target_idx = kernel->labels_.at(label_name);
          if(instr->predicate_mask_){ // conditional branch
            cbb->setEnd(i);
            tryAddSuccessor(cbb, target_idx);
            
            int following_idx = i+1;
            tryAddSuccessor(cbb, following_idx);
            break;
          } else { // unconditional branch
            cbb->setEnd(i);
            tryAddSuccessor(cbb, target_idx);
            break;
          }
        } else if(find_if(kernel->labels_.begin(), kernel->labels_.end(), 
                  [i](pair<string, int> const& kv){ return kv.second == i+1; }) != kernel->labels_.end()){
          cbb->setEnd(i);
          tryAddSuccessor(cbb, i+1);
          break;
        } else {}

      } // for (i=start_idx; ; ++i)
    }

  } // CFG::CFG(Kernel const* kernel)

  CFG::~CFG(){
    for(BasicBlock* bb : basic_blocks_) delete bb;
    for(Edge* edge : edges_) delete edge;
  }

  void CFG::printCFG() const {
    for(auto const& bb : basic_blocks_){
      cout << '[' <<bb->start_ << '-' << bb->end_ << ']'
           << "\t->\t";
      for(auto const& edge : bb->outEdges_){
        auto const& successor = edge->head_;
        cout << "\n\t\t[" << successor->start_ << '-' << successor->end_ << "]\n";             
      } 
      cout << '\n';
    }
  }
} // namespace dada
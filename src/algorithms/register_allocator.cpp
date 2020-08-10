#include <memory>
#include <iostream>
#include <algorithm> // std::set_difference

#include "register_allocator.h"

using namespace std;

namespace dada{
//----------------------RegisterAllocator------------------
  RegisterAllocator::RegisterAllocator(Kernel const* kernel, CFG const* cfg)
    : kernel_(kernel), cfg_(cfg) {}

  void RegisterAllocator::setMaxRegisters(int max_regs){
    if(max_regs > 255) throw runtime_error("maximum allowed number of registers is 255, while" + to_string(max_regs) + " is given.\n");
    max_registers_ = max_regs;
  }

  void RegisterAllocator::setMaxPredicateRegisters(int max_preg){
    max_registers_ = max_preg;
  }

  int RegisterAllocator::getRegisterCount() const {
    int reg_count = 0;
    for(auto const& [op, phys_reg] : reg_alloc_result_){
      reg_count = max(reg_count, phys_reg.index + phys_reg.width);
    }
    return reg_count;
  }

  void RegisterAllocator::printResult() const {
    if(!preg_alloc_result_.empty()){
      cout << "Allocated predicate registers:\n";
      for(auto const& [op, idx] : preg_alloc_result_){
        auto const& [name, offset] = op;
        cout << name << '\t' << offset << "\t:\t" << idx << '\n';
      }
    }
    cout << '\n';

    cout << "Allocated registers:\n";
    for(auto const& [op, phys_reg] : reg_alloc_result_){
      auto const& [name, offset] = op;
      cout << name << '\t' << offset << "\t:\t" << phys_reg.index << ',' << phys_reg.width << '\n';
    }
  }

//----------------------LinearScanAllocator----------------
  LinearScanAllocator::LinearScanAllocator(Kernel const* kernel, CFG const* cfg)
    : RegisterAllocator(kernel, cfg) {}
  void LinearScanAllocator::allocate(){
    auto liveness_info = computeRegisterLiveness();

    // cout << "Liveness info:\n";
    // for(auto [op, range] : liveness_info){
    //   auto [name, offset] = op;
    //   auto [assign, use] = range;
    //   cout << name << '\t' << offset << "\t[" << assign << '-' << use << "]\n";
    // }

    vector<pair<string, int>> ordered_ops;
    for(auto [op, range] : liveness_info) ordered_ops.push_back(op);
    sort(ordered_ops.begin(), ordered_ops.end(), [&liveness_info](
      pair<string, int> const& op1, pair<string, int> const& op2){
        return liveness_info.at(op1).first < liveness_info.at(op2).first;
    });

    set<pair<string, int>> active_ops;
    vector<bool> free_regs(max_registers_, true);
    vector<bool> free_pregs(max_predicate_registers_, true);

    for(auto const& op : ordered_ops){
      auto const [assign, use] = liveness_info.at(op);

      // 1. expire old
      for(auto const& aop : active_ops){
        // TODO: change this to 
        // liveness_info.at(aop).second <= assign ?
        if(liveness_info.at(aop).second < assign){
          active_ops.erase(aop);
          auto [name, offset] = aop;
          DataType type = kernel_->symbol_table_.at(name)->data_type_;
          if(type == PRED){
            free_pregs[preg_alloc_result_.at(aop)] = true;
          } else {
            for(int i=0; i<reg_alloc_result_.at(aop).width; ++i)
              free_regs[reg_alloc_result_.at(aop).index+i] = true;
          }
        }
      }

      // 2. try allocate new
      auto [name, offset] = op;
      
      DataType type = kernel_->symbol_table_.at(name)->data_type_;
      int vec_size  = kernel_->symbol_table_.at(name)->vector_size_;
      if(type == PRED) {
        bool is_alloc = false;
        for(int i=0; i<max_predicate_registers_; ++i){
          if(free_pregs[i] == true){ // allocate predicate register
            preg_alloc_result_[op] = i;
            free_pregs[i] = false;           
            is_alloc = true; break;
          }
        }
        if(!is_alloc) throw runtime_error(name + " cannot be allocated.\n");
      }
      else {
        int phys_reg_width = Operand::data_type_width.at(type) / 32;
        if(vec_size == 1){
          bool is_alloc = false;
          if(phys_reg_width == 1){
            for(int i=0; i<max_registers_; ++i){
              if(free_regs[i] == true){
                reg_alloc_result_[op] = PhysReg{i, 1};
                free_regs[i] = false;
                is_alloc = true; break;
              }
            }
          } else if(phys_reg_width == 2){
            for(int i=0; i<max_registers_; i+=2){
              if(free_regs[i] == true && free_regs[i+1] == true){
                reg_alloc_result_[op] = PhysReg{i, 2};
                free_regs[i] = false; free_regs[i+1] = false;
                is_alloc = true; break;
              }
            }
          } else if(phys_reg_width == 4){
            for(int i=0; i<max_registers_; i+=4){
              if(free_regs[i] == true && free_regs[i+1] == true &&
                 free_regs[i+2] == true && free_regs[i+3] == true){
                reg_alloc_result_[op] = PhysReg{i, 4};
                free_regs[i] = false; free_regs[i+1] = false;
                free_regs[i+2] = false; free_regs[i+3] = false;
                is_alloc = true; break;
              }
            }
          }
          if(!is_alloc) throw runtime_error(name + " cannot be allocated.\n");
          else active_ops.insert(op);
        } else {
          int vec_base = offset / vec_size * vec_size;
          int vec_off  = offset % vec_size;
          if(reg_alloc_result_.find(make_pair(name, vec_base)) == reg_alloc_result_.end()){
            bool is_alloc = false;
            int total_width = phys_reg_width * vec_size;
            for(int i=0; i<max_registers_; i+=total_width){
              for(int j=0; j<total_width; ++j){
                if(free_regs[i+j] == false){
                  is_alloc = false; 
                  break;
                }
                if(j == total_width-1){
                  is_alloc = true;
                  for(int k=0; k<total_width; ++k) free_regs[i+k] = false;
                  for(int v=0; v<vec_size; ++v) {
                    reg_alloc_result_[make_pair(name, vec_base+v)] = 
                                      PhysReg{i+v*phys_reg_width, phys_reg_width};
                    active_ops.insert(make_pair(name, vec_base+v));
                  }
                }
              }
              if(is_alloc) break;
            }
            if(!is_alloc) throw runtime_error(name + " cannot be allocated.\n");
          } else {
            // already allocated. do nothing
          }
        }
      }
    } // for each op

    // For debugging.
    // printResult();
    
  } // LinearScanAllocator::allocate()

  // return liveness info for each operand.
  map<pair<string, int>, pair<int, int>> LinearScanAllocator::computeRegisterLiveness() {
    struct LiveInfo {
        set<pair<string, int>> use;
        set<pair<string, int>> def;
        set<pair<string, int>> live_in;
        set<pair<string, int>> live_out;

        BasicBlock* bb_;
        Kernel const* kernel_;

        // Compute use & def
        LiveInfo(BasicBlock* bb, Kernel const* kernel) : bb_(bb), kernel_(kernel){
          using op  = pair<string, int>;
          int start = bb->start_;
          int end = bb->end_;
          for(int i = start; i<=end; ++i){
            unique_ptr<Instruction> const& instr = kernel_->instructions_[i];
            vector<unique_ptr<Operand>> const& src_operands = instr->src_operands_;
            vector<unique_ptr<Operand>> const& dst_operands = instr->dst_operands_;
            unique_ptr<Operand> const& pmask = instr->predicate_mask_;
            
            if(pmask){
              if(def.find(pmask->getOpPair()) == def.end()){
                use.insert(pmask->getOpPair());}}

            for(auto const& src : src_operands){
              if((src->type_ == ID || src->type_ == MEM_REF) && 
                  src->state_space_ == REG){
                for(op const& src_op : src->getOpPairs()){
                  if(def.find(src_op) == def.end()){ use.insert(src_op); }
                }
              }
            }

            for(auto const& dst : dst_operands){
              if(dst->type_ == ID && dst->state_space_ == REG){
                for(op const& dst_op : dst->getOpPairs()){
                  def.insert(dst_op);
                }}}

          } // for i=start...end;
        } // LiveInfo(BasicBlock* bb)

        // live_in = (live_out - def) U use
        // return if live_in has been changed.
        bool updateLiveIn(map<BasicBlock*, unique_ptr<LiveInfo>> const& live_infos){
          // update live_out first
          for(auto const& edge : bb_->outEdges_){
            auto const& succ = edge->head_;
            for(auto op : live_infos.at(succ)->live_in) live_out.insert(op);
          }
          set<pair<string, int>> new_live_in;
          set_difference(live_out.begin(), live_out.end(), 
                         def.begin(), def.end(), inserter(new_live_in, new_live_in.begin()));
          new_live_in.insert(use.begin(), use.end());
          if(new_live_in.size() == live_in.size()) return false;
          else {
            live_in = new_live_in;
            return true;
          }
        }

        // live_out = U(succ) live_in
        void updateLiveOut(map<BasicBlock*, unique_ptr<LiveInfo>> const& live_infos){
          live_out.clear();
          for(Edge const* out_edge : bb_->outEdges_){
            auto const& succ_live_info = live_infos.at(out_edge->head_);
            live_out.insert(succ_live_info->live_in.begin(), succ_live_info->live_in.end());
          }
        }
    };

    map<BasicBlock*, unique_ptr<LiveInfo>> live_infos;
    for(BasicBlock* bb : cfg_->basic_blocks_){
      live_infos[bb] = make_unique<LiveInfo>(bb, kernel_);
    }

    set<BasicBlock*> working_set (cfg_->basic_blocks_.begin(), cfg_->basic_blocks_.end());
    while(!working_set.empty()){
      BasicBlock* cbb = *(working_set.begin());
      working_set.erase(working_set.begin());

      if(live_infos[cbb]->updateLiveIn(live_infos)){
        for(Edge* edge : cbb->inEdges_){
          working_set.insert(edge->tail_);
        }
      }
    } // while(!working_set.empty())

    // Algorithm converge.
    map<pair<string, int>, pair<int, int>> result;

    struct {
      set<BasicBlock*> visited;
      map<pair<string, int>, pair<int, int>>& result;
      Kernel const* kernel_;
      map<BasicBlock*, unique_ptr<LiveInfo>>& live_infos;

      void opWrite(pair<string, int> op, int instr_idx){
        // TODO: check this.
        if(result.find(op) == result.end()){
          throw runtime_error(op.first + "\t" + to_string(op.second) + " is never used. This is considered an error.\n");
        }
        auto [assign, use] = result.at(op);
        if(instr_idx < assign || instr_idx > use) return;
        if(instr_idx > assign && instr_idx < use) {
          if(assign == -1) { result[op] = make_pair(instr_idx+1, use); return;}
          else { return; }
        }
        // last instruction of current bb.
        if(assign == -1 && use == instr_idx) result[op] = make_pair(instr_idx, instr_idx);
      }

      void opRead(pair<string, int> op, int instr_idx){
        if(result.find(op) == result.end()){
          result[op] = make_pair(-1, instr_idx);
          return;
        }
        auto [assign, use] = result.at(op);
        if(instr_idx > assign && instr_idx < use) return;
        if(instr_idx < assign) {
          result[op] = make_pair(-1, use);
          return;
        }
        if(instr_idx > use){
          result[op] = make_pair(assign, instr_idx);
          return;
        }
      }

      void operator()(BasicBlock* bb){
        visited.insert(bb);

        // live_out are alive at bb's end.
        auto const& live_info = live_infos.at(bb);
        for(auto op : live_info->live_out){
          opRead(op, bb->end_);
        }

        // scan backward
        for(int i=bb->end_; i>=bb->start_; --i){
          auto const& instr = kernel_->instructions_[i];
          // 1. dst is marked as not in use
          for(auto const& dst : instr->dst_operands_){
            if(dst->state_space_ == REG && (dst->type_ == ID || dst->type_ == MEM_REF)){
              for(pair<string, int> dst_op : dst->getOpPairs()){
                opWrite(dst_op, i);}}}

          // 2. src & pmask are marked as in use
          for(auto const& src : instr->src_operands_){
            if(src->state_space_ == REG && (src->type_ == ID || src->type_ == MEM_REF)){
              for(pair<string, int> src_op : src->getOpPairs()){
                opRead(src_op, i);}}}
          if(instr->predicate_mask_){ opRead(instr->predicate_mask_->getOpPair(), i); }
        }

        for(Edge* edge : bb->outEdges_){
          BasicBlock* succ = edge->head_;
          if(visited.find(succ) == visited.end()){            
            (*this)(succ);
          }
        }
      }

    } DFS{.result=result, .kernel_=kernel_, .live_infos=live_infos};

    DFS(cfg_->root_);

    return result;
  }

//----------------------BinPackingAllocator----------------

//----------------------GraphColoringAllocator-------------
} // namespace dada
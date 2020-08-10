#include "barrier_setter.h"
#include "src/arch/arch-common.h"

#include <memory>
#include <iostream>
#include <set>
#include <unordered_set>
#include <climits>
#include <algorithm>
#include <sstream>
#include <list>

using namespace std;

namespace dada {
//----------------------BarPair & SuperBarPair---------
  BarPair::BarPair(int s_idx, int e_idx, BarrierType b_type)
    : start_idx(s_idx), end_idx(e_idx), bar_type(b_type) {}

  bool BarPair::operator==(BarPair const& other){
    if(start_idx == other.start_idx && 
        end_idx   == other.end_idx &&
        bar_type  == other.bar_type){
      return true;
    } else {
      return false;
    }
  }

  SuperBarPair::SuperBarPair(BarPair const& bar_pair) 
    : end_idx(bar_pair.end_idx), bar_type(bar_pair.bar_type) {
    start_idxs.push_back(bar_pair.start_idx);
  }

  void SuperBarPair::merge(SuperBarPair const& other){
    start_idxs.insert(start_idxs.end(), other.start_idxs.begin(), other.start_idxs.end());
    sort(start_idxs.begin(), start_idxs.end());
    end_idx = min(end_idx, other.end_idx); 
  }

  string SuperBarPair::str() const {
    stringstream ret;
    for(int start_idx : start_idxs){
      if(!ret.str().empty()) ret << ',';
      ret << hex << start_idx;
    }
    ret << ':' << end_idx << '\t'
        << ((bar_type == BarrierType::RAW)? "RAW" : "WAR");
    return ret.str();
  }

//-----------------------BarrierSetter-----------------
  BarrierSetter::BarrierSetter(Kernel const* kernel, CFG const* cfg, 
    RegisterAllocator const* register_allocator, StallSetter const* stall_setter, int arch)
    : kernel_(kernel), cfg_(cfg), register_allocator_(register_allocator), 
      stall_setter_(stall_setter), arch_(arch) {}

  void BarrierSetter::setMaxBarriers(int max_barriers){
    max_barriers_ = max_barriers;
  }

  void BarrierSetter::set(){
    switch(alg_){
      case BarrierSettingAlg::SINGLE_BB:
        setSingleBB(); break;
      case BarrierSettingAlg::LINEAR_SCAN:
        setLinearScan(); break;
      case BarrierSettingAlg::SECOND_CHANCE:
        setSecondChance(); break;
    }
  }

  void BarrierSetter::setSingleBB() {
    unique_ptr<BaseArch> arch (makeArch(arch_));

    using op = pair<string, int>;
    // enum BarrierType {RAW, WAR};

    // 1. Find live_out registers.
    map<BasicBlock*, std::set<int>> live_outs = getLiveOuts();

    for(BasicBlock* bb : cfg_->basic_blocks_){
      // if(!bb->inEdges_.empty() && !bb->outEdges_.empty()) continue;
      vector<unique_ptr<BarPair>> bar_pairs;

      map<Operand*, int> active_RAW_ops; // op=>write_instr_idx;
      map<int, int> active_WAR_ops; // reg_idx=>read_instr_idx;

      for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
        auto const& instr = kernel_->instructions_[instr_idx];

        for(auto const& src : instr->src_operands_){
          if((src->type_ == ID || src->type_ == MEM_REF) && src->state_space_ == REG && src->data_type_ != PRED){
            for(auto iter = active_RAW_ops.begin(); iter != active_RAW_ops.end(); ++iter){
              if(iter->first->contains(src.get()) || src->contains(iter->first)){
                bar_pairs.emplace_back(new BarPair(iter->second, instr_idx, BarrierType::RAW));
                active_RAW_ops.erase(iter);
                break;
              } 
            }

            if(arch->isVariableLatency(instr.get())){
              for(auto const& src_op : src->getOpPairs()){
                auto const& src_phys_reg = register_allocator_->reg_alloc_result_.at(src_op);
                for(int src_reg_idx = src_phys_reg.index; 
                    src_reg_idx < src_phys_reg.index + src_phys_reg.width;
                    src_reg_idx++){
                  active_WAR_ops[src_reg_idx] = instr_idx;
                }
              }
            }

          }
        } // for each src

        for(auto const& dst : instr->dst_operands_){
          if((dst->type_ == ID || dst->type_ == MEM_REF) && dst->state_space_ == REG && dst->data_type_ != PRED){
            for(auto const& dst_op : dst->getOpPairs()){
              auto const& dst_phys_reg = register_allocator_->reg_alloc_result_.at(dst_op);
              for(int dst_reg_idx = dst_phys_reg.index;
                  dst_reg_idx < dst_phys_reg.index + dst_phys_reg.width;
                  dst_reg_idx++){
                for(auto iter = active_WAR_ops.begin(); iter != active_WAR_ops.end(); ++iter){
                  if(dst_reg_idx == iter->first){
                    if(iter->second != instr_idx) // e.g., ldg.64 r0, [r0]; we do not need barrier here.
                      bar_pairs.emplace_back(new BarPair(iter->second, instr_idx, BarrierType::WAR));
                    active_WAR_ops.erase(iter);
                    break;
                  }
                }
              }
            }

            if(arch->isVariableLatency(instr.get())){
              active_RAW_ops[dst.get()] = instr_idx;
            }
          }
        } // for each dst

      } // for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx)

      // the last instruction needs to wait on all active barriers.
      while(!active_RAW_ops.empty()) {
        bar_pairs.emplace_back(new BarPair(active_RAW_ops.begin()->second, bb->end_, BarrierType::RAW));
        active_RAW_ops.erase(active_RAW_ops.begin());
      }

      // if succ bbs do not write to this reg, we don't have to wait
      std::set<int> live_out_regs = live_outs.at(bb);       
      while(!active_WAR_ops.empty()){
        auto const& [read_reg_idx, read_instr_idx] = *active_WAR_ops.begin();
        if(live_out_regs.find(read_reg_idx) != live_out_regs.end()) // needs to wait it.
          bar_pairs.emplace_back(new BarPair(read_instr_idx, bb->end_, BarrierType::WAR));
        active_WAR_ops.erase(active_WAR_ops.begin());
      }
       
      sort(bar_pairs.begin(), bar_pairs.end(), [](auto const& bp0, auto const& bp1){
        return bp0->start_idx < bp1->start_idx;}
      );      

      // remove duplicates
      for(auto curr = bar_pairs.begin(); curr != bar_pairs.end(); ++curr){
        for(auto probe = curr+1; probe != bar_pairs.end();){
          if(**curr == **probe){  probe = bar_pairs.erase(probe); } 
          else if((*curr)->start_idx == (*probe)->start_idx &&
                  (*curr)->bar_type  == (*probe)->bar_type){
            if((*curr)->end_idx < (*probe)->end_idx){
              probe = bar_pairs.erase(probe);
            } else {
              (*curr).swap(*probe);
              probe = bar_pairs.erase(probe);
            }
          } else {  probe++;  }
        } // for probe
      }

      if(cost_ == BarrierMergeCost::PLAIN){
        setSingleBbPlain(bar_pairs, bb, arch);
      } else if(cost_ == BarrierMergeCost::EST_CYCLE_DIFF){
        setSingleBbEstCycleDiff2(bar_pairs, bb, arch);
      }
    } // for each basic block
  }

  void BarrierSetter::setLinearScan() {
    throw runtime_error("BarrierSetter::setLinearScan() has not been implemented.\n");
  }

  void BarrierSetter::setSecondChance() {
    throw runtime_error("BarrierSetter::setSecondChance() has not been implemented.\n");
  }

  // live_in = (live_out - use) U def
  // live_out = U(succ) live_in
  // We only care about live_out for WAR barriers.
  map<BasicBlock*, std::set<int>> BarrierSetter::getLiveOuts() const {
    struct RegLiveInfo {
      // physical register index
      std::set<int> use;
      std::set<int> def;
      std::set<int> live_in;
      std::set<int> live_out;

      BasicBlock* bb_;

      // compute use & def
      RegLiveInfo(BasicBlock* bb, Kernel const* kernel, 
                  RegisterAllocator const* register_allocator) : bb_(bb){
        for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
          auto const& instr = kernel->instructions_[instr_idx];
          auto const& src_operands_ = instr->src_operands_;
          auto const& dst_operands_ = instr->dst_operands_;

          for(auto const& src : src_operands_){
            if( (src->type_ == ID || src->type_ == MEM_REF) &&
                src->state_space_ == REG &&
                src->data_type_ != PRED ){
              for(auto const& src_op : src->getOpPairs()){
                auto const& phys_reg = register_allocator->reg_alloc_result_.at(src_op);
                for(int reg_idx = phys_reg.index; 
                    reg_idx < phys_reg.index + phys_reg.width; 
                    ++reg_idx){
                  use.insert(reg_idx);
                }
              }
            }
          } // for(auto const& src : src_operands_)

          for(auto const& dst : dst_operands_){
            if(dst->type_ == ID && dst->state_space_ == REG && dst->data_type_ != PRED){
              for(auto const& dst_op : dst->getOpPairs()){
                auto const& phys_reg = register_allocator->reg_alloc_result_.at(dst_op);
                for(int reg_idx = phys_reg.index; 
                    reg_idx < phys_reg.index + phys_reg.width;
                    ++reg_idx){
                  if(use.find(reg_idx) == use.end()){
                    def.insert(reg_idx);
                  }
                }
              }
            }
          } // for(auto const& dst : dst_operands_)
        }
      } // RegLiveInfo constructor

      // Return true if live_in has be modified
      // live_in = (live_out - use) U def
      bool updateLiveIn(map<BasicBlock*, unique_ptr<RegLiveInfo>> const& reg_live_infos){
        // update live_out first
        for(auto const& edge : bb_->outEdges_){
          auto const& succ = edge->head_;
          for(int reg_idx : reg_live_infos.at(succ)->live_in){
            live_out.insert(reg_idx);
          }
        }

        // live_in = (live_out - use) U def
        std::set<int> new_live_in;
        set_difference(live_out.begin(), live_out.end(), 
                       use.begin(), use.end(),
                       inserter(new_live_in, new_live_in.begin()));
        new_live_in.insert(def.begin(), def.end());

        if(new_live_in.size() == live_in.size()) return false;
        else {
          live_in = new_live_in;
          return true;
        }
      }

    };

    map<BasicBlock*, unique_ptr<RegLiveInfo>> live_infos_bb;
    for(BasicBlock* bb : cfg_->basic_blocks_)
      live_infos_bb[bb] = make_unique<RegLiveInfo>(bb, kernel_, register_allocator_);

    std::set<BasicBlock*> working_set(cfg_->basic_blocks_.begin(), cfg_->basic_blocks_.end());

    while(!working_set.empty()){
      BasicBlock* cbb = *working_set.begin();
      working_set.erase(cbb);

      if(live_infos_bb[cbb]->updateLiveIn(live_infos_bb)){
        for(Edge* edge : cbb->inEdges_){
          working_set.insert(edge->tail_);
        }
      }
    }

    // return value.
    map<BasicBlock*, std::set<int>> rv;
    for(auto const& [bb, reg_live_info] : live_infos_bb){
      std::set<int> live_out = reg_live_info->live_out;
      rv.emplace(bb, live_out);
    }

    return rv;
  }

  //-----------------------------SingleBB's Merge cost-------------------------
  void BarrierSetter::setSingleBbPlain(
    vector<unique_ptr<BarPair>> const& bar_pairs, BasicBlock const* bb, unique_ptr<BaseArch> const& arch){

      std::set<int> free_barriers;
      for(int i=0; i<max_barriers_; ++i) free_barriers.insert(i);

      // std::set<BarPair*> active_bar_pairs;
      vector<pair<BarPair*, int>> active_bar_pairs;
      map<BarPair*, int> bar_alloc_result;

      auto bp_iter = bar_pairs.begin();
      for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
        if(bp_iter == bar_pairs.end()) break;

        // 1. expire old
        for(auto iter = active_bar_pairs.begin(); iter != active_bar_pairs.end(); ){
          if(iter->first->end_idx <= instr_idx){
            int bar_idx = bar_alloc_result.at(iter->first);
            free_barriers.insert(bar_idx);
            iter = active_bar_pairs.erase(iter);
          } else{
            ++iter;
          }
        }

        // 2. try to allocate new
        while(bp_iter != bar_pairs.end() && (*bp_iter)->start_idx == instr_idx){
          if(free_barriers.empty()){
            auto min_remaining_iter = min_element(active_bar_pairs.begin(), active_bar_pairs.end(), 
                                [](pair<BarPair*, int> const& lhs, pair<BarPair*, int> const& rhs){
                                    return lhs.second < rhs.second; });
            min_remaining_iter->first->end_idx = instr_idx;
            int bar_idx = bar_alloc_result.at(min_remaining_iter->first);
            free_barriers.insert(bar_idx);
            // cout << "No free barrier. Force barrier " << bar_idx 
            //      << " to wait at instr " << instr_idx << '\n';
          } 
          
          int free_bar_idx = *free_barriers.begin();
          bar_alloc_result.emplace(bp_iter->get(), free_bar_idx);
          active_bar_pairs.emplace_back(bp_iter->get(), 
                                        arch->getLatency(kernel_->instructions_[instr_idx].get()));
          free_barriers.erase(free_bar_idx);

          ++bp_iter;         
        }

        int stall_cycles = stall_setter_->stalls_[instr_idx];
        for(auto& [sbp, remaining] : active_bar_pairs){
          remaining -= stall_cycles;
        }
      }

      cout << "Barrier setting result:\n";
      vector<pair<BarPair*, int>> bar_alloc_result_sort(
        bar_alloc_result.begin(), bar_alloc_result.end());
      sort(bar_alloc_result_sort.begin(), bar_alloc_result_sort.end(), 
       [](pair<BarPair*, int> const& lhs, pair<BarPair*, int> const& rhs){ 
         return lhs.first->start_idx < rhs.first->start_idx;});

      for(auto const& [bar_pair, bar_idx] : bar_alloc_result_sort){
        cout << hex << bar_pair->start_idx << ':' << bar_pair->end_idx << '\t'
             << (bar_pair->bar_type == BarrierType::RAW ? "RAW" : "WAR") << "\t:\t" << bar_idx << '\n';
      }


      // bar_alloc_result ==> real result (output)
      for(auto const& [bar_pair, bar_idx] : bar_alloc_result){
        if(bar_pair->bar_type == BarrierType::RAW){
          write_barriers_.emplace(bar_pair->start_idx, bar_idx);
        } else {
          read_barriers_.emplace(bar_pair->start_idx, bar_idx);
        }

        if(wait_barriers_.find(bar_pair->end_idx) == wait_barriers_.end()){
          vector<int> waits{bar_idx};
          wait_barriers_.emplace(bar_pair->end_idx, waits);
        } else {
          wait_barriers_[bar_pair->end_idx].push_back(bar_idx);
        }
      }
  }

  void BarrierSetter::setSingleBbEstCycleDiff(
    vector<unique_ptr<BarPair>> const& bar_pairs, 
    BasicBlock const* bb, unique_ptr<BaseArch> const& arch
  ) {
    // Change bar_pair to super_bar
    vector<unique_ptr<SuperBarPair>> superbar_pairs;
    for(auto const& bar_pair : bar_pairs){
      superbar_pairs.emplace_back(new SuperBarPair(*bar_pair));
    }

    std::set<int> free_barriers;
    for(int i=0; i<max_barriers_; ++i) free_barriers.insert(i);

    std::set<SuperBarPair*> active_super_bar_pairs;
    map<SuperBarPair*, int> super_bar_alloc_result;

    auto sbp = superbar_pairs.begin();
    for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
      if(sbp == superbar_pairs.end()) break;

      // 1. expire old
      for(auto iter = active_super_bar_pairs.begin(); 
          iter != active_super_bar_pairs.end(); ){
        if((*iter)->end_idx <= instr_idx){
          int bar_idx = super_bar_alloc_result.at(*iter);
          free_barriers.insert(bar_idx);
          // cout << "At instr " << instr_idx << ", "
          //      << "barrier " << (*iter)->str() << " is expired. Barrier " << bar_idx << " is free.\n";
          iter = active_super_bar_pairs.erase(iter);
        } else {
          ++iter;
        }
      } // expire old (for active_super_bar_pairs)

      // 2. try to allocate new
      while(sbp != superbar_pairs.end() && (*sbp)->start_idxs[0] == instr_idx){
        active_super_bar_pairs.insert(sbp->get());

        if(free_barriers.empty()){
          // reset allocated & free barriers
          for(int i=0; i<max_barriers_; ++i) free_barriers.insert(i);
          for(SuperBarPair* active_sbp : active_super_bar_pairs){
            super_bar_alloc_result.erase(active_sbp);
          }

          // compute pairwise cost
          auto [lhs, rhs] = getMinMergeCostPair(active_super_bar_pairs, arch);
          // cout << "No free barrier. Merge "
          //      << lhs->str() << " and "  << rhs->str() << '\n';
          lhs->merge(*rhs);
          active_super_bar_pairs.erase(rhs);

          // allocate barriers.
          for(SuperBarPair* active_sbp : active_super_bar_pairs){
            int free_bar_idx = *free_barriers.begin();
            super_bar_alloc_result[active_sbp] = free_bar_idx;
            free_barriers.erase(free_bar_idx);
            // cout << "(Merged)At instr " << instr_idx << ", "
            //      << "allocate " << free_bar_idx << " to " << active_sbp->str() << '\n';
          }

        } else {
          int free_bar_idx = *free_barriers.begin();
          super_bar_alloc_result[sbp->get()] = free_bar_idx;
          free_barriers.erase(free_bar_idx);
          // cout << "At instr " << instr_idx << ", "
          //      << "allocate " << free_bar_idx << " to " << (*sbp)->str() << '\n';
        }
        ++sbp;
      }
    } // instr_idx = bb->start_ ... bb->end_;

    // // print debug info
    // cout << "Barrier setting result:\n";
    // vector<pair<SuperBarPair*, int>> super_bar_alloc_result_sort(
    //   super_bar_alloc_result.begin(), super_bar_alloc_result.end());
    // sort(super_bar_alloc_result_sort.begin(), super_bar_alloc_result_sort.end(), 
    //   [](pair<SuperBarPair*, int> const& lhs, pair<SuperBarPair*, int> const& rhs){ 
    //     return lhs.first->start_idxs[0] < rhs.first->start_idxs[0];});

    // for(auto const& [bar_pair, bar_idx] : super_bar_alloc_result_sort){
    //   cout << hex << bar_pair->str() << "\t" << bar_idx << '\n';
    // }

    // super_bar_alloc_result ==> real result
    for(auto& [super_bar_pair, bar_idx] : super_bar_alloc_result){
      if(super_bar_pair->bar_type == BarrierType::RAW){
        for(int write_idx : super_bar_pair->start_idxs){
          write_barriers_.emplace(write_idx, bar_idx);
        }
      } else if(super_bar_pair->bar_type == BarrierType::WAR){
        for(int read_idx : super_bar_pair->start_idxs){
          read_barriers_.emplace(read_idx, bar_idx);
        }
      }

      if(wait_barriers_.find(super_bar_pair->end_idx) == wait_barriers_.end()){
        vector<int> waits{bar_idx};
        wait_barriers_.emplace(super_bar_pair->end_idx, waits);
      } else {
        wait_barriers_[super_bar_pair->end_idx].push_back(bar_idx);
      }
    }

  }

  void BarrierSetter::setSingleBbEstCycleDiff2(
    vector<unique_ptr<BarPair>> const& bar_pairs, 
    BasicBlock const* bb, unique_ptr<BaseArch> const& arch
  ) {
    // Change bar_pair to super_bar
    vector<unique_ptr<SuperBarPair>> superbar_pairs;
    for(auto const& bar_pair : bar_pairs){
      superbar_pairs.emplace_back(new SuperBarPair(*bar_pair));
    }

    // cout << "Raw barriers:\n";
    // for(auto const& super_bar : superbar_pairs){
    //   cout << super_bar->str() << endl;
    // }

    // count max #of concurrent_barriers
    vector<list<SuperBarPair*>> concurrent_barriers(bb->end_ - bb->start_ + 1);
    int max_concurr = 0;
    for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
      int i = instr_idx - bb->start_;
      for(auto const& super_bar : superbar_pairs){
        if(super_bar->start_idxs[0] <= instr_idx && super_bar->end_idx > instr_idx){
          concurrent_barriers[i].push_back(super_bar.get());
        } else if(super_bar->end_idx == bb->end_ && instr_idx == bb->end_){ // treat last instr specially
          concurrent_barriers[i].push_back(super_bar.get());
        }
      }
      int curr_concurr = concurrent_barriers[i].size();
      max_concurr = max(curr_concurr, max_concurr);
    }

    // cout << "max concurr: " << max_concurr << endl;
    // cout << "For each instr:\n";
    // for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
    //   cout << hex << instr_idx << ":" << concurrent_barriers[instr_idx - bb->start_].size() << "\n";
    //   for(auto const& super_bar : concurrent_barriers[instr_idx - bb->start_]){
    //     cout << '\t' << super_bar->str() << '\t';
    //   }
    //   cout << '\n';
    // }

    while(max_concurr > max_barriers_){
      auto iter = find_if(concurrent_barriers.begin(), concurrent_barriers.end(), 
                          [this](auto const& list_sbp){ return list_sbp.size() > max_barriers_;});
      // cout << (iter - concurrent_barriers.begin()) + bb->start_ << '\n';
      std::set<SuperBarPair*> active_super_bar_pairs(iter->begin(), iter->end());
      auto [lhs, rhs] = getMinMergeCostPair(active_super_bar_pairs, arch);

      // cout << "Merge:\n"
      //      << lhs->str() << " and\t" << rhs->str() << endl;
      // merge
      if(lhs->end_idx <= *(rhs->start_idxs.rbegin())){
        // Split rhs
        for(auto iter = rhs->start_idxs.begin(); iter != rhs->start_idxs.end(); ){
          if(*iter < lhs->end_idx){
            lhs->start_idxs.push_back(*iter);
            iter = rhs->start_idxs.erase(iter);
          } else {
            ++iter;
          }
        }
        sort(lhs->start_idxs.begin(), lhs->start_idxs.end());
      } else if(rhs->end_idx <= *(lhs->start_idxs.rbegin())){
        // Split lhs
        for(auto iter = lhs->start_idxs.begin(); iter != lhs->start_idxs.end(); ){
          if(*iter < rhs->end_idx){
            rhs->start_idxs.push_back(*iter);
            iter = lhs->start_idxs.erase(iter);
          } else {
            ++iter;
          }
        }
        sort(rhs->start_idxs.begin(), rhs->start_idxs.end());
      } else {
        // merge two & delete one
        lhs->merge(*rhs);
        for(auto iter = superbar_pairs.begin(); iter != superbar_pairs.end(); ++iter){
          if(iter->get() == rhs){
            superbar_pairs.erase(iter);
            break;
          }
        }
      }

      // recompute max_concurr
      concurrent_barriers = vector<list<SuperBarPair*>>(bb->end_ - bb->start_ + 1);
      max_concurr = 0;
      for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
        int i = instr_idx - bb->start_;
        for(auto const& super_bar : superbar_pairs){
          if(super_bar->start_idxs[0] <= instr_idx && super_bar->end_idx > instr_idx){
            concurrent_barriers[i].push_back(super_bar.get());
          } else if(super_bar->end_idx == bb->end_ && instr_idx == bb->end_){ // treat last instr specially
            concurrent_barriers[i].push_back(super_bar.get());
          }
        }
        int curr_concurr = concurrent_barriers[i].size();
        max_concurr = max(curr_concurr, max_concurr);
      }
    }

    // cout << "(After merge) max concurr: " << max_concurr << endl;
    // cout << "For each instr:\n";
    // for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
    //   cout << hex << instr_idx << ":" << concurrent_barriers[instr_idx - bb->start_].size() << "\n";
    //   for(auto const& super_bar : concurrent_barriers[instr_idx - bb->start_]){
    //     cout << '\t' << super_bar->str() << '\t';
    //   }
    //   cout << '\n';
    // }

    // allocate real barriers
    std::set<int> free_barriers;
    for(int i=0; i<max_barriers_; ++i) free_barriers.insert(i);

    std::set<SuperBarPair*> active_super_bar_pairs;
    map<SuperBarPair*, int> super_bar_alloc_result;

    sort(superbar_pairs.begin(), superbar_pairs.end(), 
      [](auto const& sbp0, auto const& sbp1){ 
        return sbp0->start_idxs[0] < sbp1->start_idxs[0];
      });

    auto sbp = superbar_pairs.begin();
    for(int instr_idx = bb->start_; instr_idx <= bb->end_; ++instr_idx){
      if(sbp == superbar_pairs.end()) break;

      // 1. expire old
      for(auto iter = active_super_bar_pairs.begin(); iter != active_super_bar_pairs.end(); ){
        if((*iter)->end_idx <= instr_idx){
          int bar_idx = super_bar_alloc_result.at(*iter);
          free_barriers.insert(bar_idx);
          // cout << "At instr " << instr_idx << ", "
          //      << "barrier " << (*iter)->str() << " is expired. Barrier " << bar_idx << " is free.\n";
          iter = active_super_bar_pairs.erase(iter);
        } else {
          ++iter;
        }
      } // expire old (for active_super_bar_pairs)

      // 2. allocate new
      while(sbp != superbar_pairs.end() && (*sbp)->start_idxs[0] == instr_idx){
        active_super_bar_pairs.insert(sbp->get());

        int free_bar_idx = *free_barriers.begin();
        super_bar_alloc_result[sbp->get()] = free_bar_idx;
        free_barriers.erase(free_bar_idx);

        ++sbp;
      }
    } // for bb->start_ ... bb->end_

    // // print debug info
    // cout << "Barrier setting result:\n";
    // vector<pair<SuperBarPair*, int>> super_bar_alloc_result_sort(
    //   super_bar_alloc_result.begin(), super_bar_alloc_result.end());
    // sort(super_bar_alloc_result_sort.begin(), super_bar_alloc_result_sort.end(), 
    //   [](pair<SuperBarPair*, int> const& lhs, pair<SuperBarPair*, int> const& rhs){ 
    //     return lhs.first->start_idxs[0] < rhs.first->start_idxs[0];});

    // for(auto const& [bar_pair, bar_idx] : super_bar_alloc_result_sort){
    //   cout << hex << bar_pair->str() << "\t" << bar_idx << '\n';
    // }

    // super_bar_alloc_result ==> real result
    for(auto& [super_bar_pair, bar_idx] : super_bar_alloc_result){
      if(super_bar_pair->bar_type == BarrierType::RAW){
        for(int write_idx : super_bar_pair->start_idxs){
          write_barriers_.emplace(write_idx, bar_idx);
        }
      } else if(super_bar_pair->bar_type == BarrierType::WAR){
        for(int read_idx : super_bar_pair->start_idxs){
          read_barriers_.emplace(read_idx, bar_idx);
        }
      }

      if(wait_barriers_.find(super_bar_pair->end_idx) == wait_barriers_.end()){
        vector<int> waits{bar_idx};
        wait_barriers_.emplace(super_bar_pair->end_idx, waits);
      } else {
        wait_barriers_[super_bar_pair->end_idx].push_back(bar_idx);
      }
    }
  }

  pair<SuperBarPair*, SuperBarPair*> BarrierSetter::getMinMergeCostPair(
    std::set<SuperBarPair*> active_super_bar_pairs, //
    unique_ptr<BaseArch> const& arch
  ){
    int min_merge_cost = INT_MAX;
    pair<SuperBarPair*, SuperBarPair*> min_merge_cost_pair;
    for(auto iter_a = active_super_bar_pairs.begin(); 
        iter_a != active_super_bar_pairs.end(); ++iter_a){
      for(auto iter_b = next(iter_a); iter_b != active_super_bar_pairs.end(); ++iter_b){
        int curr_merge_cost = getMergeCost(*iter_a, *iter_b, arch);
        if(curr_merge_cost < min_merge_cost) {
          min_merge_cost = curr_merge_cost;
          min_merge_cost_pair = make_pair(*iter_a, *iter_b);
        }
      }
    }

    return min_merge_cost_pair;
  }

  int BarrierSetter::getMergeCost(
    SuperBarPair const* lhs, SuperBarPair const* rhs, unique_ptr<BaseArch> const& arch) {
    if(lhs->bar_type != rhs->bar_type) return INT_MAX;

    int original_cycles = 0;
    int p_begin = min(lhs->start_idxs[0], rhs->start_idxs[0]);
    int p_end = max(lhs->end_idx, rhs->end_idx);

    auto lhs_start_iter = lhs->start_idxs.begin();
    auto rhs_start_iter = rhs->start_idxs.begin();

    // 1. compute original_cycles
    int lhs_remaining = 0;
    int rhs_remaining = 0;
    for(int instr_idx = p_begin; instr_idx <= p_end; ++instr_idx){
      if(lhs_start_iter != lhs->start_idxs.end() && *lhs_start_iter == instr_idx){
        lhs_remaining = max(lhs_remaining, 
                            arch->getLatency(kernel_->instructions_[instr_idx].get()));
      }
      if(rhs_start_iter != rhs->start_idxs.end() && *rhs_start_iter == instr_idx){
        rhs_remaining = max(rhs_remaining, 
                            arch->getLatency(kernel_->instructions_[instr_idx].get()));
      }
      
      if(lhs->end_idx == instr_idx && rhs->end_idx != instr_idx){
        int waiting_cycles = max(lhs_remaining, 0);
        rhs_remaining -= waiting_cycles;
        original_cycles += waiting_cycles;
      } else if(rhs->end_idx == instr_idx && lhs->end_idx != instr_idx){
        int waiting_cycles = max(rhs_remaining, 0);
        lhs_remaining -= waiting_cycles;
        original_cycles += waiting_cycles;
      } else if(rhs->end_idx == lhs->end_idx){
        int waiting_cycles = max(max(rhs_remaining, lhs_remaining), 0);
        original_cycles += waiting_cycles;
      }

      int current_stalls = stall_setter_->stalls_[instr_idx];
      if(instr_idx != p_end){
        lhs_remaining -= current_stalls;
        rhs_remaining -= current_stalls;
        original_cycles += current_stalls;
      }
    }

    // 2. compute merged_cycles
    int merged_cycles = 0;
    int merge_remaining = 0;
    int m_end = min(lhs->end_idx, rhs->end_idx);
    lhs_start_iter = lhs->start_idxs.begin();
    rhs_start_iter = rhs->start_idxs.begin();
    for(int instr_idx = p_begin; instr_idx <= p_end; ++instr_idx){
      if(lhs_start_iter != lhs->start_idxs.end() && *lhs_start_iter == instr_idx){
        merge_remaining = max(merge_remaining, 
                              arch->getLatency(kernel_->instructions_[instr_idx].get()));
      }
      if(rhs_start_iter != rhs->start_idxs.end() && *rhs_start_iter == instr_idx){
        merge_remaining = max(merge_remaining, 
                              arch->getLatency(kernel_->instructions_[instr_idx].get()));
      }

      if(instr_idx == m_end){
        int waiting_cycles = max(merge_remaining, 0);
        merged_cycles += waiting_cycles;
      }

      int current_stalls = stall_setter_->stalls_[instr_idx];
      if(instr_idx != p_end){
        merge_remaining -= current_stalls;
        merged_cycles += current_stalls;
      }
    }

    // 3. return cost
    // cout << "Cost for merge " << lhs->str() << " and " << rhs->str() << " is "
    //      << merged_cycles - original_cycles << '\n';
    return merged_cycles - original_cycles;
  }
}
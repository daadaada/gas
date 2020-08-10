#pragma once

#include <map>
#include <vector>

#include "src/core/kernel.h"
#include "cfg.h"
#include "register_allocator.h"
#include "stall_setter.h"
#include "src/arch/arch-common.h"

namespace dada {
  enum class BarrierSettingAlg {
    SINGLE_BB,
    LINEAR_SCAN,
    SECOND_CHANCE,
  };

  enum class BarrierMergeCost {
    PLAIN,
    NUM_INSTR,
    EST_CYCLE_DIFF,
  };

  enum class BarrierType {
    RAW, WAR,
  };

  struct BarPair {
    BarPair(int s_idx, int e_idx, BarrierType b_type);
    bool operator==(BarPair const& other);
    int start_idx = -1;
    int end_idx = -1;
    BarrierType bar_type = BarrierType::RAW;
  };

  // Represents merged barriers
  struct SuperBarPair {
    SuperBarPair(BarPair const& bar_pair);
    std::string str() const;
    std::vector<int> start_idxs;
    int end_idx = -1;
    BarrierType bar_type = BarrierType::RAW;
    void merge(SuperBarPair const& other);
  };

  class BarrierSetter {
    private:
      Kernel const* kernel_;
      CFG const* cfg_;
      RegisterAllocator const* register_allocator_;
      StallSetter const* stall_setter_;
      int const arch_ = 0;

      int max_barriers_ = 6;
      BarrierSettingAlg alg_ = BarrierSettingAlg::SINGLE_BB;
      BarrierMergeCost cost_ = BarrierMergeCost::EST_CYCLE_DIFF;
      // BarrierMergeCost cost_ = BarrierMergeCost::PLAIN;

    public:
      // result
      std::map<int, int> read_barriers_;
      std::map<int, int> write_barriers_;
      std::map<int, std::vector<int>> wait_barriers_;

    public:
      BarrierSetter(
        Kernel const* kernel, CFG const* cfg, RegisterAllocator const*, 
        StallSetter const* stall_setter, int arch);

      void set();

      void setMaxBarriers(int max_barriers);
      void setAlgorithm(BarrierSettingAlg, BarrierMergeCost);
    
    private:
      void setSingleBB();
      void setLinearScan();
      void setSecondChance();

      void setSingleBbPlain(
        std::vector<std::unique_ptr<BarPair>> const& bar_pairs, 
        BasicBlock const* bb, std::unique_ptr<BaseArch> const& arch
      );
      void setSingleBbEstCycleDiff(
        std::vector<std::unique_ptr<BarPair>> const& bar_pairs, 
        BasicBlock const* bb, std::unique_ptr<BaseArch> const& arch
      );
      void setSingleBbEstCycleDiff2(
        std::vector<std::unique_ptr<BarPair>> const& bar_pairs, 
        BasicBlock const* bb, std::unique_ptr<BaseArch> const& arch
      );

      std::pair<SuperBarPair*, SuperBarPair*> getMinMergeCostPair(
        std::set<SuperBarPair*> active_super_bar_pairs, 
        std::unique_ptr<BaseArch> const& arch
      );

      int getMergeCost(
        SuperBarPair const* lhs, SuperBarPair const* rhs,
        std::unique_ptr<BaseArch> const& arch
      );

    private:
      std::map<BasicBlock*, std::set<int>> getLiveOuts() const;
  };
}
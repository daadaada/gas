#pragma once

#include "arch_sm7x.h"

namespace dada {
  class SM75 : public SM7X {
    public:
      bool isVariableLatency(Instruction const* instr) override;
      int getLatency(Instruction const* instr) override;
      int getTputLatency(Instruction const* instr1, Instruction const* instr2) override;
      int getParameterBaseOffset() const override;

      std::vector<uint64_t> getInstructionBinary(
        Instruction const*, RegisterAllocator const*, CtrlLogic const&) const override;

    private:
    //--------------------------------Latency-info-------------------------------------
      enum class Resource {
        FP16, FP32, FP64, SFU,
        TC, LDST, INT, OTHER,
      };

      static inline const std::map<Opcode, int> latency = {
        {HADD2, 7}, {HMUL2, 7}, {HFMA2, 7}, // variable?
        {FADD, 5}, {FMUL, 5}, {FFMA, 5}, {MUFU, 4},
        {DADD, 4}, {DMUL, 4}, {DFMA, 4},
        {HMMA, 14}, {IMMA, 14}, {BMMA, 14}, // To be confirmed
        {LDG, 2000}, {STG, 20}, {LDS, 30}, {STS, 20}, {LDC, 8}, // LDC?
        {MOV, 6}, {SHFL, 20},
        {IADD3, 5}, {IMAD, 5}, {LEA, 5},
        {ISETP, 12}, {PSETP, 12},
        {SHF, 8}, {LOP3, 8},
        {IMNMX, 5}, {FMNMX, 5}, 
        {I2I, 5}, {I2F, 5}, {F2I, 5}, {F2F, 5}, // To be confirmed
        {BRA, 5}, {EXIT, 5}, {JMP, 5},
        {P2R, 12}, {R2P, 12},
        {BAR, 5}, {CS2R, 5}, {NOP, 1}, {S2R, 20},
      };

      static inline const std::set<Opcode> variable_latency = {
        MUFU, DADD, DMUL, DFMA, 
        LDG, STG, LDS, STS, LDC,
        SHFL, S2R,
        DADD, DMUL, DFMA,
      };

      static inline const std::map<Opcode, std::map<Flag, int>> special_latency = {
        {IMAD, {{WIDE, 8}}}, 
        // HMMA, I2F, etc.
      };

      static inline const std::map<Opcode, Resource> resource_type = {
        {HADD2, Resource::FP16}, {HMUL2, Resource::FP16}, {HFMA2, Resource::FP16}, // To be confirmed
        {FADD, Resource::FP32}, {FMUL, Resource::FP32}, {FFMA, Resource::FP32}, {MUFU, Resource::SFU},
        {DADD, Resource::FP64}, {DMUL, Resource::FP64}, {DFMA, Resource::FP64},
        {HMMA, Resource::TC}, {IMMA, Resource::TC}, {BMMA, Resource::TC},
        {LDG, Resource::LDST}, {STG, Resource::LDST}, {LDS, Resource::LDST}, {STS, Resource::LDST}, 
        {LDC, Resource::LDST},
        {MOV, Resource::INT}, {SHFL, Resource::LDST}, // To be confirmed
        {IADD3, Resource::INT}, {IMAD, Resource::INT}, {LEA, Resource::INT},
        {ISETP, Resource::INT}, {PSETP, Resource::INT}, // To be confirmed (PSETP)
        {SHF, Resource::INT}, {LOP3, Resource::INT}, 
        {IMNMX, Resource::INT}, {FMNMX, Resource::FP32},
        {I2I, Resource::INT}, {I2F, Resource::INT}, {F2I, Resource::FP32}, {F2I, Resource::FP32},
        {BRA, Resource::OTHER}, {EXIT, Resource::OTHER}, {JMP, Resource::OTHER},
        {P2R, Resource::INT}, {R2P, Resource::INT},
        {BAR, Resource::OTHER}, {CS2R, Resource::OTHER}, {NOP, Resource::OTHER}, {S2R, Resource::LDST},
      };

      static inline const std::map<Opcode, int> op_tput = {
        {HADD2, 2}, {HMUL2, 2}, {HFMA2, 2},
        {FADD, 2}, {FMUL, 2}, {FFMA, 2}, {MUFU, 8},
        {DADD, 8}, {DMUL, 8}, {DFMA, 8},
        {HMMA, 4}, {IMMA, 4}, {BMMA, 4},
        {LDG, 8}, {STG, 8}, {LDS, 4}, {STS, 4}, 
        {LDC, 4},
        {MOV, 2}, {SHFL, 4}, // To be confirmed
        {IADD3, 2}, {IMAD, 2}, {LEA, 2},
        {ISETP, 2}, {PSETP, 2}, // To be confirmed (PSETP)
        {SHF, 2}, {LOP3, 2}, 
        {IMNMX, 2}, {FMNMX, 2},
        {I2I, 2}, {I2F, 2}, {F2I, 2}, {F2I, 2},
        {BRA, 5}, {EXIT, 2}, {JMP, 2},
        {P2R, 4}, {R2P, 4},
        {BAR, 5}, {CS2R, 1}, {NOP, 1}, {S2R, 2},
      };

    //-------------------------------Encoding-info--------------------------------------
      static inline const std::map<Opcode, uint64_t> op_encoding_ = {
        {HADD2, 0x230}, {HFMA2, 0x231}, {HMUL2, 0x232}, 
        {FADD,  0x221}, {FFMA,  0x223}, {FMUL,  0x220}, {MUFU, 0x308}, 
        {DADD,  0x229}, {DMUL,  0x228}, {DFMA,  0x22b}, 
        {HMMA, 0x23c}, {IMMA, 0x237}, {BMMA, 0x23d},
        {LDG, 0x381}, {STG, 0x386}, {LDS, 0x984}, {STS, 0x388}, {LDC, 0xb82}, 
        {MOV,  0x202}, {SHFL, 0x0}, // SHFL
        {IADD3, 0x210}, {IMAD,  0x224}, {LEA, 0x211},
        {ISETP, 0x20c}, {PSETP, 0x0}, // ISETP, PSETP?
        {SHF,  0x219},  {LOP3,  0x212},       
        {IMNMX, 0x0}, {FMNMX, 0x0},
        {I2I, 0x0}, {I2F, 0x0}, {F2I, 0x0}, {F2F, 0x0}, // To be confirmed.
        {BRA, 0x947}, {EXIT,  0x94d}, {JMP, 0x0}, // To be confirmed. Important. 
        {P2R, 0x803}, {R2P, 0x804},
        {BAR, 0xb1d}, {CS2R, 0x805}, {NOP, 0x918}, {S2R, 0x919},
      };

      static inline const std::map<Opcode, std::map<Flag, uint64_t>> alter_op_encoding_ = {
        {IMAD, {{WIDE, 0x225}, /*{HI, 0x227}*/}}, 
        // {HMMA, {{HMMA1688, 0x23c}}}, 
        
      };

      static inline const std::map<Flag, uint64_t> flag_encoding_ = {
        {RN, 0x0}, {RM, 0x4000}, {RP, 0x8000}, {RZ, 0xc000}, 
        {TRUNC, 0x0}, {FLOOR, 0x0}, {CEIL, 0x0},
        {HMMA1688, 0x0}, {IMMA8816, 0x0}, {IMMA8832, 56<<16}, {BMMA88128, 0x0},
        {WIDE, 0x0},
        {X_FLAG, 1<<10},
        {EQ, 2<<12}, {NE, 5<<12}, {LT, 1<<12}, {LE, 3<<12}, {GT, 4<<12}, {GE, 6<<12},
        {EQU, 0x0}, {NEU, 0x0}, {LEU, 0x0}, {GTU, 0x0}, {GEU, 0x0}, {NAN, 0x0}, {NUM, 0x0},
        {LDST16, 2<<9}, {LDST32, 4<<9}, {LDST64, 5<<9}, {LDST128, 6<<9},
        {COS, 0x0}, {SIN, 1<<10}, {EX2, 0x0}, {LG2, 0x0}, {RCP, 0x0}, {RSQ, 0x0},
        {IS32, 0x200}, {IU32, 0x0},
        {AND, 0x0}, {XOR, 0x0}, {OR, 0x0},
        {CTA, 0x0}, {GPU, 2<<13}, {SYS, 3<<13},
        {MCONST, 0x0}, {STRONG_DEFAULT, 1<<15}, {MSTRONG, 2<<15}, {MWEAK, 3<<15},
        {EF, 0x0}, {CACHE_DEFAULT, 1<<20}, {EL, 2<<20}, {LU, 3<<20},
        {SHIFT_L, 0x0}, {SHIFT_R, 1<<12},
        {SYNC, 0x0},
        {LUT, 0x0},
        {F16T, 0x0}, {F32T, 1<<12},
      };

      static inline const std::map<Flag, std::map<Opcode, uint64_t>> alter_flag_encoding_ = {
        // {CS2R, {{LDST32, 0}, {LDST64, 0}}},
      };

      static inline uint32_t const parameter_base_offset = 0x160;

    //------------------------------Helper-functions------------------------------------
    private:
      uint64_t getOpEncoding(Instruction const* instr) const; 
      uint64_t getFlagsEncoding(Instruction const* instr) const;

      void modifyICR(Opcode, uint64_t& first, Operand const* src, int loc) const ;
  };
}
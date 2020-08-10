#pragma once

#include <vector>
#include <string>
#include <memory>

#include "operand.h"

namespace dada {
  enum Opcode {
    HADD2, HMUL2, HFMA2,
    FADD,  FMUL,  FFMA, MUFU,
    DADD,  DMUL,  DFMA,
    HMMA,  IMMA,  BMMA, 
    LDG,   STG,   LDS,  STS,  LDC, 
    MOV,   SHFL,
    IADD3, IMAD,  LEA,
    ISETP, PSETP, // FSETP
    SHF,   LOP3,  // PLOP3
    IMNMX, FMNMX,
    I2I, I2F, F2I, F2F,
    BRA, EXIT, JMP,
    P2R, R2P,
    BAR, CS2R, NOP, S2R,
  };
  
  enum Flag {
    RN, RZ, RM, RP,
    TRUNC, FLOOR, CEIL,
    HMMA1688, IMMA8816, IMMA8832, BMMA88128,
    WIDE,
    X_FLAG,
    EQ, NE, LT, LE, GT, GE, 
    EQU, NEU, LEU, GTU, GEU, NAN, NUM,
    LDST16, LDST32, LDST64, LDST128,
    COS, SIN, EX2, LG2, RCP, RSQ, 
    IS32, IU32, 
    AND, XOR, OR, 
    CTA, GPU, SYS,
    MCONST, MWEAK, MSTRONG, STRONG_DEFAULT,
    EF, EL, LU, CACHE_DEFAULT,
    SHIFT_L, SHIFT_R, 
    SYNC,
    LUT,
    F16T, F32T, S32T, U32T, S8T, U8T, S4T,  U4T,
  };

  class Instruction {
    public:
      Opcode opcode_;
      std::vector<Flag> flags_;
      std::vector<std::unique_ptr<Operand>> operands_;
      std::unique_ptr<Operand> predicate_mask_;

      std::vector<std::unique_ptr<Operand>> src_operands_;
      std::vector<std::unique_ptr<Operand>> dst_operands_;

      uint32_t source_line_no_ = 0;

    public:
      // constructors
      Instruction(Opcode, std::vector<Flag>, std::vector<Operand*>);
      Instruction(Operand* pred_mask, Opcode, std::vector<Flag>, std::vector<Operand*>);

      // observers
      void checkSemantic();
    
    public:
      static inline std::map<std::string, Opcode> const str_to_opcode = {
        {"hadd2", HADD2}, {"hfma2", HFMA2}, {"hmul2", HMUL2},
        {"fadd", FADD}, {"ffma", FFMA}, {"fmul", FMUL}, {"mufu", MUFU},
        {"dadd", DADD}, {"dfma", DFMA}, {"dmul", DMUL},
        {"hmma", HMMA}, {"imma", IMMA}, {"bmma", BMMA},
        {"ldg", LDG}, {"lds", LDS}, {"ldc", LDC}, {"stg", STG}, {"sts", STS}, 
        {"mov", MOV}, {"shfl", SHFL},
        {"iadd3", IADD3}, {"imad", IMAD}, {"lea", LEA},
        {"isetp", ISETP}, {"psetp", PSETP},        
        {"shf", SHF}, {"lop3", LOP3}, 
        {"imnmx", IMNMX}, {"fmnmx", FMNMX},
        {"i2i", I2I}, {"i2f", I2F}, {"f2i", F2I}, {"f2f", F2F},
        {"bra", BRA}, {"exit", EXIT}, {"jmp", JMP},
        {"p2r", P2R}, {"r2p", R2P},
        {"bar", BAR}, {"cs2r", CS2R}, {"nop", NOP}, {"s2r", S2R},
      };

      static inline std::map<std::string, Flag> const str_to_flag = {
        {".rn", RN}, {".rz", RZ}, {".rm", RM}, {".rp", RP},
        {".trunc", TRUNC}, {".floor", FLOOR}, {".ceil", CEIL},
        {".1688", HMMA1688}, {".8816", IMMA8816}, {".8832", IMMA8832}, {".88128", BMMA88128},
        {".wide", WIDE},
        {".x", X_FLAG},
        {".eq", EQ}, {".ne", NE}, {".lt", LT}, {".gt", GT}, {".ge", GE},
        {".equ", EQU}, {".neu", NEU}, {".leu", LEU}, {".gtu", GTU}, {".geu", GEU}, {".nan", NAN}, {".num", NUM},
        {".16", LDST16}, {".32", LDST32}, {".64", LDST64}, {".128", LDST128},
        {".cos", COS}, {".sin", SIN}, {".ex2", EX2}, {".lg2", LG2}, {".rcp", RCP}, {".rsq", RSQ},
        {".and", AND}, {".xor", XOR}, {".or", OR},
        {".cta", CTA}, {".gpu", GPU}, {".sys", SYS},
        {".constant", MCONST}, {".week",  MWEAK}, {".strong", MSTRONG},
        {".ef", EF}, {".el", EL}, {".lu", LU},
        {".l", SHIFT_L}, {".r", SHIFT_R},
        {".sync", SYNC},
        {".lut", LUT},
        {".f16", F16T}, {".f32", F32T}, {".s32", S32T}, {".u32", U32T},
        {".s8", S8T}, {".u8", U8T}, {".s4", S4T}, {".u4", U4T},
      };

    private:
      // helpers for semantic checking.
      void checkFloatFlags();
      void checkGmemFlags();
      void checkSmemFlags();
      int getLdstWidth() const;


  };
} // namespace dada
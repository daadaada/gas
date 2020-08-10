#pragma once

#include <map>
#include <set>
#include <vector>

namespace dada {
  enum OperandType {
    ID, MEM_REF, FIMM, IIMM, LABEL,
  };
  enum StateSpace {
    REG, 
    CREG, // rz, pt 
    SREG, // tidx, tidy etc.
    CONST, // params
  };
  enum DataType {
    B16X2, B32, B64,
    U16X2, U32, U64, 
    S16X2, S32, S64,
    F16X2, F32, F64,
    PRED,
  };

  class MRef {
    private:
      std::string name_;
      int mem_offset_;
      friend class Operand;
    public:
      MRef(std::string name, int mem_offset);
  };

  class Operand{
    public:
      std::string name_;
      int iimm_ = 0;
      float fimm_ = 0.0f;
      int offset_ = -1; // for array. E.g., a[0]; -1 for scalar variable
      int vector_length_ = 1; // E.g., a[0:3]. length=4;
      int mem_offset_ = 0; // for memory offset. E.g., ldg a, [a_ptr+0x200];
      int label_offset_ = -1; // for label only.
      int param_offset_ = 0; // for param(const) only. offset in bytes.
      OperandType type_;
      StateSpace state_space_;
      DataType data_type_;
      bool is_neg_ = false; // for ID only
      bool is_abs_ = false; // for ID only

    public:
      Operand(int value);
      Operand(float value);
      Operand(std::string name, bool is_neg);
      Operand(std::string name, int offset, bool is_neg);
      Operand(std::string name, int offset, int vector_length, bool is_neg);
      Operand(MRef* mref);

    public:
      int getBitWidth() const;

      std::pair<std::string, int> getOpPair() const;
      std::vector<std::pair<std::string, int>>  getOpPairs() const;

    public:
      bool contains(Operand const*) const;

    public:
     static inline std::map<std::string, DataType> const str_to_data_type = {
       {"b16x2", B16X2}, {"b32", B32}, {"b64", B64},
       {"u16x2", U16X2}, {"u32", U32}, {"u64", U64},
       {"s16x2", S16X2}, {"s32", S32}, {"s64", S64},
       {"f16x2", F16X2}, {"f32", F32}, {"f64", F64},
       {"pred", PRED},
     };

     static inline std::map<DataType, int> const data_type_width = {
       {B16X2, 32}, {B32, 32}, {B64, 64},
       {U16X2, 32}, {U32, 32}, {U64, 64},
       {S16X2, 32}, {S32, 32}, {S64, 64},
       {F16X2, 32}, {F32, 32}, {F64, 64},
       {PRED, 1},
     };

     static inline std::set<std::string> const constant_registers = {
       "rz", "pt",
     };

     static inline std::set<std::string> const special_registers = {
       "threadIdx_x", "threadIdx_y", "threadIdx_z", 
       "blockIdx_x", "blockIdx_y", "blockIdx_z",
       "srz", "clock_lo",
     };

     static inline std::set<std::string> const predefined_constant = {
       "blockDim_x", "blockDim_y", "blockDim_z",
       "gridDim_x", "gridDim_y", "gridDim_z",
     };

  };
} // namespace dada
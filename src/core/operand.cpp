#include "operand.h"

using namespace std;

namespace dada {
//-----------------------MRef--------------------
  MRef::MRef(string name, int mem_offset)
    : name_(name), mem_offset_(mem_offset) {}

//-----------------------Operand-----------------
  Operand::Operand(int value) : iimm_(value){
    type_ = IIMM;
    state_space_ = REG; 
  }

  Operand::Operand(float value) : fimm_(value){
    type_ = FIMM;
    state_space_ = REG;
  }

  Operand::Operand(string name, bool is_neg) : name_(name), is_neg_(is_neg){
    type_ = ID;
    // Check for SREG & CREG & CONST
    if(constant_registers.find(name_) != constant_registers.end()){
      state_space_ = CREG;
      if(name == "pt") data_type_ = PRED;
    } else if(special_registers.find(name_) != special_registers.end()){
      state_space_ = SREG;
    } else if(predefined_constant.find(name_) != predefined_constant.end()){
      state_space_ = CONST;
    } else{
      state_space_ = REG;
    }
  }

  Operand::Operand(string name, int offset, bool is_neg)
    : name_(name), offset_(offset), is_neg_(is_neg){
    type_ = ID;
    if(constant_registers.find(name_) != constant_registers.end()){
      state_space_ = CREG;
    } else {
      state_space_ = REG;
    }
  }

  Operand::Operand(string name, int offset, int vector_length, bool is_neg)
    : name_(name), offset_(offset), vector_length_(vector_length), is_neg_(is_neg){
    type_ = ID;
    if(constant_registers.find(name_) != constant_registers.end()){
      state_space_ = CREG;
    } else {
      state_space_ = REG;
    }
  }

  Operand::Operand(MRef* mref){
    name_ = mref->name_;
    mem_offset_ = mref->mem_offset_;
    type_ = MEM_REF;
    if(constant_registers.find(name_) != constant_registers.end()){
      state_space_ = CREG;
    } else {
      state_space_ = REG;
    }
  }

  //-------------------Operand-other-----------------------
  int Operand::getBitWidth() const {
    if(type_ != ID && type_ != MEM_REF)
      return 0; // only ID has bit width
    if(state_space_ == CREG && name_ == "rz")
      return 0;
    return data_type_width.at(data_type_) * vector_length_;
  }

  pair<string, int> Operand::getOpPair() const {
    return make_pair(name_, offset_);
  }

  vector<pair<string, int>> Operand::getOpPairs() const {
    vector<pair<string, int>> op_paris;
    for(int i=0; i<vector_length_; ++i) op_paris.push_back(make_pair(name_, offset_+i));
    return op_paris;
  }

  bool Operand::contains(Operand const* other) const{
    if(!((type_ == ID || type_ == MEM_REF) && state_space_ == REG))
      return false;
    if(!((other->type_ == ID || other->type_ == MEM_REF) && other->state_space_ == REG))
      return false;

    if(name_ == other->name_ &&
       offset_ <= other->offset_ &&
       offset_ + vector_length_ >= other->offset_ + other->vector_length_)
      return true;
    else 
      return false;
  }

} // namespace dada
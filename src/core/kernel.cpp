#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "kernel.h"

using namespace std;

namespace dada{
//------------------Parameter--------------------
  Parameter::Parameter(string name, DataType type)
    : name_(name), data_type_(type){
    size_ = Operand::data_type_width.at(data_type_);
  }

  int Parameter::getSize() const {
    return size_;
  }

  void Parameter::setOffset(int offset){
    offset_ = offset;
  }

//------------------Symbol-----------------------
  Symbol::Symbol(string name, DataType type, int array_size)
    : name_(name), data_type_(type), array_size_(array_size) {}

  Symbol::Symbol(string name, DataType type, int array_size, int vector_size)
    : name_(name), data_type_(type), array_size_(array_size), vector_size_(vector_size) {}

//------------------Statement--------------------
  Statement::Statement(Instruction* instr) : instr_(instr){
    is_instruction_ = true;
  }

  Statement::Statement(vector<Symbol*> symbols) : symbols_(symbols){
    is_symbol_ = true;
  }

  Statement::Statement(string label_name) : label_name_(label_name){
    is_label_ = true;
  }

  bool Statement::isInstruction() const { return is_instruction_; }
  bool Statement::isSymbol() const { return is_symbol_; }
  bool Statement::isLabel() const { return is_label_; }

//------------------Kernel-----------------------
  Kernel::Kernel(string name) : name_(name){}

  void Kernel::addParameters(vector<Parameter*> params){
    int offset = 0;
    for(Parameter* param : params) {
      param->setOffset(offset);
      parameters_.emplace_back(param);
      offset += param->getSize()/8;
    }
  }

  void Kernel::addStatements(vector<Statement*> stmts){
    // record label location:
    int instr_size = 0;
    for(Statement* stmt : stmts){
      if(stmt->isInstruction()){ instr_size++; }
      else if(stmt->isLabel()){
        string label_name = stmt->label_name_;
        if(labels_.find(label_name) != labels_.end()){
          throw runtime_error(label_name + " has be defined as label already.\n");
        } else{
          labels_[label_name] = instr_size;
        }
      }
    }

    // Lookup symbols & check semantic
    for(Statement* stmt : stmts){
      if(stmt->isInstruction()){
        symbolTableLookUp(stmt->instr_, instructions_.size());
        try{
          stmt->instr_->checkSemantic();
        } catch (exception const& e){
          cout << e.what() << ", at line " << stmt->instr_->source_line_no_ << "\n\n";
          throw runtime_error("Fatal error. Terminate.\n");
        }
        instructions_.emplace_back(stmt->instr_);
      } else if(stmt->isSymbol()){
        for(auto& symbol : stmt->symbols_){
          string name = symbol->name_;
          if(Operand::constant_registers.find(name) != Operand::constant_registers.end() ||
             Operand::special_registers.find(name) != Operand::special_registers.end() || 
             Operand::predefined_constant.find(name) != Operand::predefined_constant.end()){
               throw runtime_error(name + " is a reserved name.\n");
          } else if(symbol_table_.find(name) != symbol_table_.end()){
            throw runtime_error(name + " has be defined in " + name_);
          } else if(any_of(parameters_.begin(), parameters_.end(), 
          [name](unique_ptr<Parameter> const& param){return param->name_==name;})){
            throw runtime_error(name + " is a parameter name. But defined as variable\n");
          } else {
            symbol_table_[name] = unique_ptr<Symbol>(symbol);
          }          
        }
      } 
    } // for each statement.

  }

  string const& Kernel::getName() const {
    return name_;
  }
  
  //----------------Kernel-private--------------
  void Kernel::symbolTableLookUp(Instruction* instr, int instr_idx){
    for(auto const& op : instr->operands_){
      if(op->type_ == ID || op->type_ == MEM_REF){
        if(op->state_space_ == REG){
          if(symbol_table_.find(op->name_) != symbol_table_.end()){
            op->data_type_ = symbol_table_.at(op->name_)->data_type_;
            if(op->offset_ != -1){ // for array
              if(op->offset_ + op->vector_length_ > symbol_table_.at(op->name_)->array_size_){
                throw runtime_error(op->name_ + "'s offset is out of bound.\n");
              }
              if(op->vector_length_ != 1){
                if(op->offset_ % symbol_table_.at(op->name_)->vector_size_ != 0 ||
                   op->vector_length_ > symbol_table_.at(op->name_)->vector_size_){
                  throw runtime_error(op->name_ + "'s offset is misaligned.\n");
                }
              } 
            }
          } else if(any_of(parameters_.begin(), parameters_.end(), 
                   [&op](unique_ptr<Parameter> const& param){return op->name_ == param->name_;})){
            op->state_space_ = CONST;
            for(auto const& param : parameters_) {
              if(param->name_ == op->name_){
                op->data_type_ = param->data_type_;
                op->param_offset_ = param->offset_;
                break;
              }
            }
          } else if(labels_.find(op->name_) != labels_.end()){
            op->type_ = LABEL;
            op->label_offset_ = labels_.at(op->name_) - (instr_idx + 1);
          } else {
            throw runtime_error(op->name_ + " has not been declared.\n");
          }
        }
      } 
    }
  } // symbolTableLookUp

} // namespace dada
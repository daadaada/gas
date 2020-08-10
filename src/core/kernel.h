#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "operand.h"
#include "instruction.h"

namespace dada {
  class Parameter {
    public:
      std::string name_;
      DataType data_type_;
      int size_;
      int offset_;

    public:
      Parameter(std::string name, DataType data_type);

      int getSize() const;

      void setOffset(int offset);
  };

  class Symbol {
    public:
      std::string const name_;
      DataType const data_type_;
      int const array_size_ = -1;
      int const vector_size_ = 1;
    public:
      Symbol(std::string name, DataType type, int array_size);
      Symbol(std::string name, DataType type, int array_size, int vector_size);
  };

  class Statement{
    public:
      Instruction* instr_ = nullptr;
      std::vector<Symbol*> symbols_;
      std::string label_name_;

      bool is_instruction_ = false;
      bool is_symbol_ = false;
      bool is_label_ = false;
      
    public:
      Statement(Instruction*);
      Statement(std::vector<Symbol*>); // variable
      Statement(std::string); // label

      bool isInstruction() const;
      bool isSymbol() const;
      bool isLabel() const;
  };

  class Kernel {
    public:
      std::string const name_;
      std::vector<std::unique_ptr<Parameter>> parameters_;
      std::vector<std::unique_ptr<Instruction>> instructions_;
      std::map<std::string, std::unique_ptr<Symbol>> symbol_table_;
      std::map<std::string, int> labels_;

    public:
      Kernel(std::string kernel_name);

      void addParameters(std::vector<Parameter*> params);
      void addStatements(std::vector<Statement*> stmts);

      std::string const& getName() const;
      

    private:      
      void symbolTableLookUp(Instruction*, int);


  };
} // namespace dada
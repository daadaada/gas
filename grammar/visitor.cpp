#include <string>
#include <tuple>

#include "visitor.h"

using namespace std;
using namespace dada;
using namespace antlrcpp;

Any MyVisitor::visitTranslation_unit(dadaParser::Translation_unitContext* ctx)  {
  for(auto& kernel_defination_ctx : ctx->kernel_defination()){
    module->addKernel(visit(kernel_defination_ctx).as<Kernel*>());
  }
  return 0;
}

// Kernel
Any MyVisitor::visitKernel_defination(dadaParser::Kernel_definationContext* ctx)  {
  string name = ctx->ID()->getText();
  auto params = visit(ctx->parameter_list()).as<vector<Parameter*>>();
  auto stmts  = visit(ctx->compound_statement()).as<vector<Statement*>>();
  auto kernel = new Kernel(name);
  kernel->addParameters(params);
  kernel->addStatements(stmts);
  return kernel;
}

Any MyVisitor::visitParameter_list(dadaParser::Parameter_listContext* ctx)  {
  vector<Parameter*> params;
  if(ctx->parameter_list()){
    auto result = visit(ctx->parameter_list()).as<vector<Parameter*>>();
    params.insert(params.end(), result.begin(), result.end());
  }
  params.push_back(visit(ctx->parameter()).as<Parameter*>());
  return params;
}

Any MyVisitor::visitParameter(dadaParser::ParameterContext* ctx)  {
  auto type = visit(ctx->type_specifier()).as<DataType>();
  string name = ctx->ID()->getText();
  Parameter* param = new Parameter(name, type);
  return param;
}

Any MyVisitor::visitType_specifier(dadaParser::Type_specifierContext* ctx)  {
  return Operand::str_to_data_type.at(ctx->start->getText());
}

Any MyVisitor::visitCompound_statement(dadaParser::Compound_statementContext* ctx)  {
  vector<Statement*> stmts;
  for(auto& stmt_ctx : ctx->statement()){
    stmts.push_back(visit(stmt_ctx).as<Statement*>());
  }
  return stmts;
}

Any MyVisitor::visitStatement(dadaParser::StatementContext* ctx)  {
  Statement* stmt;
  if(ctx->var_declaration()){
    stmt = new Statement(visit(ctx->var_declaration()).as<vector<Symbol*>>());
  } else if(ctx->instruction()){
    stmt = new Statement(visit(ctx->instruction()).as<Instruction*>());
  } else if(ctx->label()){
    stmt = new Statement(ctx->label()->start->getText());
  }
  return stmt;
}

Any MyVisitor::visitVar_declaration(dadaParser::Var_declarationContext* ctx)  {
  vector<Symbol*> symbols;
  auto type = visit(ctx->type_specifier()).as<DataType>();
  int vector_length = 1;
  if(ctx->vector_length()){
    vector_length = visit(ctx->vector_length()).as<int>();
  }
  for(auto const& [name, array_size] : 
      visit(ctx->declare_operand_list()).as<map<string,int>>()){
    auto symbol = new Symbol(name, type, array_size, vector_length);
    symbols.push_back(symbol);
  }
  return symbols;
}

Any MyVisitor::visitVector_length(dadaParser::Vector_lengthContext* ctx) {
  if(ctx->V2()) return 2;
  else if(ctx->V4()) return 4;
}

Any MyVisitor::visitInstruction(dadaParser::InstructionContext* ctx)  {
  Operand* pred_mask = nullptr;
  if(ctx->predicate_mask()){
    pred_mask = visit(ctx->predicate_mask()).as<Operand*>();
  }
  Opcode opcode = Instruction::str_to_opcode.at(ctx->opcode()->start->getText());

  vector<Flag> flags;
  for(auto& flag_ctx : ctx->flag()){
    flags.push_back(Instruction::str_to_flag.at(flag_ctx->start->getText()));
  }

  vector<Operand*> operands;
  if(ctx->operand_list()){
    operands = visit(ctx->operand_list()).as<vector<Operand*>>();
  }
  
  Instruction* instr;
  if(pred_mask == nullptr){instr = new Instruction(opcode, flags, operands);} 
  else { instr = new Instruction(pred_mask, opcode, flags, operands); }
  instr->source_line_no_ = ctx->SEMI()->getSymbol()->getLine();
  return instr;
}

Any MyVisitor::visitDeclare_operand_list(dadaParser::Declare_operand_listContext* ctx)  {
  map<string, int> ops_name;
  if(ctx->declare_operand_list()){
    auto result = visit(ctx->declare_operand_list()).as<map<string,int>>();
    ops_name.insert(result.begin(), result.end());
  }
  ops_name.emplace(visit(ctx->declare_operand()).as<pair<string,int>>());
  return ops_name;
}

Any MyVisitor::visitDeclare_operand(dadaParser::Declare_operandContext* ctx)  {
  pair<string, int> op_name;
  op_name.first = ctx->ID()->getText();
  op_name.second = -1;
  if(ctx->CONSTANT()){
    op_name.second = stoi(ctx->CONSTANT()->getText());
  }
  return op_name;
}

// Instructions
Any MyVisitor::visitPredicate_mask(dadaParser::Predicate_maskContext* ctx)  {
  Operand* pred_mask;
  bool is_neg = false;
  if(ctx->NOT()) is_neg = true;
  string name = ctx->ID()->getText();
  pred_mask = new Operand(name, is_neg);
  return pred_mask;
}

Any MyVisitor::visitOperand_list(dadaParser::Operand_listContext* ctx)  {
  vector<Operand*> operands;
  if(ctx->operand_list()){
    auto result = visit(ctx->operand_list()).as<vector<Operand*>>();
    operands.insert(operands.end(), result.begin(), result.end());
  }
  operands.push_back(visit(ctx->operand()).as<Operand*>());
  return operands;
}

Any MyVisitor::visitOperand(dadaParser::OperandContext* ctx)  {
  // ID | ID[const] | ID[const:const] | const | float | mem
  Operand* operand;
  bool is_neg = false;
  if(ctx->NEG()) is_neg = true;
  if(ctx->NOT()) is_neg = true;
  if(ctx->ID()){
    string name = ctx->ID()->getText();
    if(ctx->CONSTANT(0)){
      if(ctx->CONSTANT(1)){
        int start = stoi(ctx->CONSTANT(0)->getText());
        int end = stoi(ctx->CONSTANT(1)->getText());
        operand = new Operand(name, start, end-start+1, is_neg);
      } else {
        int offset = stoi(ctx->CONSTANT(0)->getText());
        operand = new Operand(name, offset, is_neg);
      }
    } else {
      operand = new Operand(name, is_neg);
    }
  } else if(ctx->CONSTANT(0)){
    int value = stoi(ctx->CONSTANT(0)->getText());
    if(is_neg) value *= -1;
    operand = new Operand(value);
  } else if(ctx->FLOAT_CONSTANT()){
    operand = new Operand(stof(ctx->FLOAT_CONSTANT()->getText()));
  } else if(ctx->memory_operand()){
    operand = new Operand(visit(ctx->memory_operand()).as<MRef*>());
  }
  return operand;
}

Any MyVisitor::visitMemory_operand(dadaParser::Memory_operandContext* ctx)  {
  MRef* mref;
  int mem_offset = 0;
  string name = ctx->ID()->getText();
  bool is_neg = false;
  if(ctx->NEG()) is_neg = true;
  if(ctx->CONSTANT()) mem_offset = stoi(ctx->CONSTANT()->getText());
  if(is_neg) mem_offset *= -1;
  mref = new MRef(name, mem_offset);
  return mref;
}

// empty functions
Any MyVisitor::visitLabel(dadaParser::LabelContext* ctx){
  return visitChildren(ctx);
}
Any MyVisitor::visitOpcode(dadaParser::OpcodeContext* ctx){
  return visitChildren(ctx);
}
Any MyVisitor::visitFlag(dadaParser::FlagContext* ctx){
  return visitChildren(ctx);
} 

#include <memory>

#include "generated/dadaLexer.h"
#include "generated/dadaParser.h"
#include "generated/dadaBaseVisitor.h"

#include "src/dada-common.h"

class MyVisitor : public dadaBaseVisitor {
  public:
    std::unique_ptr<dada::Module> module = std::make_unique<dada::Module>();

    antlrcpp::Any visitType_specifier(dadaParser::Type_specifierContext *context) override;

    antlrcpp::Any visitMemory_operand(dadaParser::Memory_operandContext *context) override;

    antlrcpp::Any visitOperand(dadaParser::OperandContext *context) override;

    antlrcpp::Any visitDeclare_operand_list(dadaParser::Declare_operand_listContext *context) override;

    antlrcpp::Any visitDeclare_operand(dadaParser::Declare_operandContext *context) override;

    antlrcpp::Any visitTranslation_unit(dadaParser::Translation_unitContext *context) override;

    antlrcpp::Any visitCompound_statement(dadaParser::Compound_statementContext *context) override;

    antlrcpp::Any visitLabel(dadaParser::LabelContext *context) override;

    antlrcpp::Any visitStatement(dadaParser::StatementContext *context) override;

    antlrcpp::Any visitVar_declaration(dadaParser::Var_declarationContext *context) override;

    antlrcpp::Any visitVector_length(dadaParser::Vector_lengthContext *context) override;

    antlrcpp::Any visitPredicate_mask(dadaParser::Predicate_maskContext *context) override;

    antlrcpp::Any visitInstruction(dadaParser::InstructionContext *context) override;

    antlrcpp::Any visitOperand_list(dadaParser::Operand_listContext *context) override;

    antlrcpp::Any visitKernel_defination(dadaParser::Kernel_definationContext *context) override;

    antlrcpp::Any visitParameter_list(dadaParser::Parameter_listContext *context) override;

    antlrcpp::Any visitParameter(dadaParser::ParameterContext *context) override;

    antlrcpp::Any visitOpcode(dadaParser::OpcodeContext *context) override;

    antlrcpp::Any visitFlag(dadaParser::FlagContext *context) override;

};
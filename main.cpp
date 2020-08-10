#include <string>
#include <set>
#include <memory>

#include "thirdparty/argparse/argparse.hpp"
#include "thirdparty/antlr4-runtime/src/antlr4-runtime.h"
#include "grammar/generated/dadaLexer.h"
#include "grammar/generated/dadaParser.h"
#include "grammar/generated/dadaBaseVisitor.h"
#include "grammar/visitor.h"

using namespace std;

int main(int argc, char *argv[]){
  argparse::ArgumentParser program("dada");

  program.add_argument("input")
    .help("Input source file path.");
  program.add_argument("-o")
    .default_value(string("a.cubin"))
    .help("Output file path.");
  program.add_argument("-arch")
    .help("Target GPU architecture.")
    .default_value(75)
    .action([](std::string const& value){
      set<string> const supported_archs = {"sm_70", "sm_75"};
      if(supported_archs.find(value) != supported_archs.end()){
        string arch_str = value.substr(value.find_last_not_of("0123456789")+1);
        return stoi(arch_str);
      } else {
        throw runtime_error(value + " is not a supported arch.\n");
      }
    });

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err){
    cout << err.what() << endl;
    return 0;
  }

  string input_path(program.get<string>("input"));
  ifstream input_stream(input_path);
  string output_path(program.get<string>("-o"));
  if(!input_stream){
    cout << "File " << input_path << " does not exists.\n";
  } else {
    antlr4::ANTLRInputStream input(input_stream);
    dadaLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    dadaParser parser(&tokens);
    antlr4::tree::ParseTree* tree = parser.translation_unit();
    // std::cout << tree->toStringTree(&parser) << std::endl; // For debugging

    MyVisitor visitor;

    try{
      visitor.visit(tree);
    } catch(exception const& e){
      cout << e.what() << endl;
      return 0;
    }

    int arch = program.get<int>("-arch");

    try{
      dada::CuElf cu_elf(visitor.module.get(), arch);
      cu_elf.set();
      cu_elf.toCubin(output_path);
    } catch(exception const& e){
      cout << e.what() << endl;
    }
    
  } 

  return 0;
}
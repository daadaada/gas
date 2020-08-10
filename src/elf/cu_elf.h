#pragma once

#include <string>
#include <memory>

#include "elf.h"
#include "src/core/module.h"
#include "src/arch/arch-common.h"

#include "src/algorithms/cfg.h"
#include "src/algorithms/register_allocator.h"
#include "src/algorithms/stall_setter.h"
#include "src/algorithms/barrier_setter.h"
#include "src/algorithms/reuse_setter.h"


namespace dada {
  struct ElfKernel {
    std::string name_;
    std::vector<char> data_;
    int num_regs_ = 0;
    int num_bars_ = 0;
    int smem_size_ = 0;
    int const_size_ = 0;
    std::vector<uint32_t> exit_offsets_;
    std::vector<uint32_t> param_sizes_;

    ElfKernel(
      std::string name, 
      std::vector<char> data,
      int num_regs,
      int num_bars,
      int smem_size,
      int const_size,
      std::vector<uint32_t> exit_offsets,
      std::vector<uint32_t> param_sizes
    ) : name_(name), data_(data), num_regs_(num_regs), smem_size_(smem_size), 
        const_size_(const_size), exit_offsets_(exit_offsets), param_sizes_(param_sizes){}
  };

  class CuElf {
    private:
      int const arch_ = 0;
      Module const* module_;
      int const cuda_version_ = 101;

      Elf64_Ehdr header_;
      std::vector<std::pair<Elf64_Shdr, std::vector<char>>> sections_;
      std::vector<Elf64_Phdr> program_headers_;

      std::vector<std::unique_ptr<ElfKernel>> elf_kernels_;

    public:
      CuElf(Module const* module, int arch);

      void set(); // Create data and store to elf_kernel_
      void toCubin(std::string output_path);

    // helper functions
    private:
      void initElfHeader();
      void generateElfKernels();
      void packCtrlLogic(CtrlLogic&, int instr_idx, Opcode, StallSetter const*, BarrierSetter const*, ReuseSetter const*);
      void setElfData();

      void addShstrtab();
      void addStrtab();
      void addSymtab();
      void addNvInfo();

      void addNvInfoN(ElfKernel const*);
      void addConst0(ElfKernel const*);
      void addText(ElfKernel const*);

      void setShstrtab();
      void setSymtab();
      void setNvInfo();
      void setNvInfoN();
      void setConst0();

      void updateSizeOffset();

      void addProgramHeaders();

      void setHeader();

    // helper data for building ELF data structures
    private:
      std::vector<std::string> section_names_;
      std::map<std::string, int> section_idx_;
      int shstrtab_idx = 0;
      uint64_t section_data_size_ = 0;

      std::map<std::string, int> text_section_idx_;

      // SymbolTable
      std::vector<Elf64_Sym> symbols_;
      std::map<std::string, int> symbol_func_idx_;
      std::vector<std::string> symbol_names_;

      
  };
}
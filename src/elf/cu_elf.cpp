#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric> // reduce, exclusive_scan

#include "cu_elf.h"

using namespace std;

namespace dada {
  CuElf::CuElf(Module const* module, int arch) : module_(module), arch_(arch) {}

  void CuElf::set() {
      initElfHeader();
      generateElfKernels();
      setElfData();
  }

  void CuElf::toCubin(string path) {
    ofstream f;
    f.open(path, ios::binary);
    if(f){
      f.write(reinterpret_cast<char const*>(&header_), sizeof(header_));

      for(auto const& [header, data] : sections_){
        f.write(data.data(), data.size());
      }

      for(auto const& [header, data] : sections_){
        f.write(reinterpret_cast<char const*>(&header), sizeof(header));
      }

      for(auto const& p_header : program_headers_){
        f.write(reinterpret_cast<char const*>(&p_header), sizeof(p_header));
      }

    }
  }

//-----------------------------Helper-functions----------------------
  void CuElf::initElfHeader() {
    // Magic number
    header_.e_ident[0] = '\x7f';
    header_.e_ident[1] = 'E';
    header_.e_ident[2] = 'L';
    header_.e_ident[3] = 'F';
    header_.e_ident[4] = '\x02';
    header_.e_ident[5] = '\x01';
    header_.e_ident[6] = '\x01';
    header_.e_ident[7] = '\x33';
    header_.e_ident[8] = '\x7';
    for(int i=9; i<EI_NIDENT; ++i) { header_.e_ident[i] = 0;}


    // Begin with empty section
    Elf64_Shdr null_shdr;
    vector<char> empty_vec;
    sections_.emplace_back(make_pair(null_shdr, empty_vec));
    section_names_.emplace_back();

    header_.e_flags |= arch_;
    header_.e_flags |= arch_ << 16;
  }

  void CuElf::generateElfKernels(){
    unique_ptr<BaseArch> arch(makeArch(arch_));
    elf_kernels_.reserve(module_->kernels_.size());

    for(auto const& kernel : module_->kernels_){
      auto cfg = make_unique<CFG>(kernel.get());

      // TODO: to support polymorphism
      unique_ptr<LinearScanAllocator> register_allocator{nullptr};
      try{
        register_allocator.reset(new LinearScanAllocator(kernel.get(), cfg.get()));
        register_allocator->allocate();
      } catch(exception const& e){
        cout << "Error in register allocator.\n"
             << e.what() << endl;
        throw runtime_error("Fatal error. Terminate.\n");
      }

      unique_ptr<StallSetter> stall_setter{nullptr};
      try{
        stall_setter.reset(new StallSetter(kernel.get(), cfg.get(), arch_));
        stall_setter->set();
      } catch(exception const& e){
        cout << "Error in stall setter.\n" << e.what() << endl;
        throw runtime_error("Fatal error. Terminate.\n");
      }

      unique_ptr<BarrierSetter> barrier_setter{nullptr};
      try{
        barrier_setter.reset(
          new BarrierSetter(
            kernel.get(), cfg.get(), register_allocator.get(), stall_setter.get(), arch_)
        );
        barrier_setter->set();
      } catch(exception const& e){
        cout << "Error in barrier setter.\n" << e.what() << endl;
        throw runtime_error("Fatal error. Terminate.\n");
      }

      unique_ptr<ReuseSetter> reuse_setter{nullptr};
      try{
        reuse_setter.reset(new ReuseSetter(kernel.get(), cfg.get(), register_allocator.get(), arch_));
        reuse_setter->set();
      } catch(exception const& e){
        cout << "Error in reuse setter.\n" << e.what() << endl;
        throw runtime_error("Fatal error. Terminate.\n");
      }

      vector<char> code_binary;
      vector<uint32_t> exit_offsets;

      for(int instr_idx = 0; instr_idx < kernel->instructions_.size(); ++instr_idx){
        auto const& instr = kernel->instructions_[instr_idx];
        if(instr->opcode_ == EXIT){
          exit_offsets.push_back(code_binary.size());
        }

        CtrlLogic ctrl_logic;
        packCtrlLogic(ctrl_logic, instr_idx, instr->opcode_, stall_setter.get(), barrier_setter.get(), reuse_setter.get());
        
        vector<uint64_t> instr_data;
        try{
          instr_data = arch->getInstructionBinary(instr.get(), register_allocator.get(), ctrl_logic);
        } catch(exception const& e){
          cout << "Error in getInstructionBinary.\n"
               << e.what() << endl;
        }

        for(uint64_t& uint64_data : instr_data){
          code_binary.insert(code_binary.end(), reinterpret_cast<char*>(&uint64_data), 
                                                reinterpret_cast<char*>(&uint64_data)+sizeof(uint64_data));
        } 
      } // for each instruction

      vector<uint32_t> param_sizes;
      int const_size = 0;
      for(auto const& param : kernel->parameters_){
        param_sizes.push_back(param->getSize()/8); // size in byte
        const_size += param->getSize()/8;
      }      
      const_size += arch->getParameterBaseOffset();

      elf_kernels_.emplace_back(new ElfKernel(
        kernel->name_,
        code_binary,
        register_allocator->getRegisterCount() + 2, // must +2. otherwise illegal instruction error will be raised.
        /*num_bars*/0,                              // at least for sm_75
        /*smem_size*/0,
        const_size,
        exit_offsets,
        param_sizes
      ));
    } // for each kernel
  }

  void CuElf::packCtrlLogic(
    CtrlLogic& ctrl_logic, int instr_idx, 
    Opcode opcode, 
    StallSetter const* stall_setter, 
    BarrierSetter const* barrier_setter, 
    ReuseSetter const* reuse_setter){
    ctrl_logic.stalls = stall_setter->stalls_[instr_idx];
    if(barrier_setter->read_barriers_.find(instr_idx) != 
       barrier_setter->read_barriers_.end()){
      ctrl_logic.read_barrier_idx = barrier_setter->read_barriers_.at(instr_idx);
    }
    if(barrier_setter->write_barriers_.find(instr_idx) != 
       barrier_setter->write_barriers_.end()){
      ctrl_logic.write_barrier_idx = barrier_setter->write_barriers_.at(instr_idx);
    }
    if(barrier_setter->wait_barriers_.find(instr_idx) != 
       barrier_setter->wait_barriers_.end()){
      vector<int> const& bs = barrier_setter->wait_barriers_.at(instr_idx);
      for(int b : bs){
        ctrl_logic.wait_barriers_mask |= 1<<b;
      }
    }

    ctrl_logic.reuse_mask = reuse_setter->reuse_masks_[instr_idx];
    // if(opcode == LDG) ctrl_logic.yield = true;
    // if(instr_idx % 8 == 7) ctrl_logic.yield = true;
  }

  void CuElf::setElfData(){
    // 1. Create shstrtab, strtab, symtab
    addShstrtab();
    addStrtab();
    addSymtab();
    
    // 2. Create .nv.info. section
    addNvInfo();

    // 3. For each kernel, create corresponding sections
    //   a. .nv.info. (required)
    //   b. .nv.info.{name} (required)
    //   c. .nv.constant0.{name} (required)
    //   d. .text.{name} (required, instruction info)
    //   e. .nv.shared.{name} (optional, static smem size)
    for(auto const& elf_kernel : elf_kernels_) addNvInfoN(elf_kernel.get());
    for(auto const& elf_kernel : elf_kernels_) addConst0(elf_kernel.get());
    for(auto const& elf_kernel : elf_kernels_) addText(elf_kernel.get());

    setShstrtab();
    setSymtab();

    setNvInfo();
    setNvInfoN();
    setConst0();

    updateSizeOffset();

    addProgramHeaders();

    setHeader();
  }

  void CuElf::addShstrtab() {
    header_.e_shstrndx = static_cast<uint16_t>(sections_.size()); 
    section_idx_[".shstrtab"] = static_cast<uint16_t>(sections_.size()); 
    Elf64_Shdr shstrtab_shdr {.sh_type=SHT_STRTAB, .sh_addralign=1};
    vector<char> empty_vec;
    sections_.emplace_back(make_pair(shstrtab_shdr, empty_vec));
    section_names_.emplace_back(".shstrtab");
  }

  void CuElf::addStrtab() {
    section_idx_[".strtab"] = static_cast<uint16_t>(sections_.size()); 
    Elf64_Shdr shstrtab_shdr {.sh_type=SHT_STRTAB, .sh_addralign=1};
    vector<char> empty_vec;
    sections_.emplace_back(make_pair(shstrtab_shdr, empty_vec));
    section_names_.emplace_back(".strtab");
  }

  void CuElf::addSymtab() {
    section_idx_[".symtab"] = static_cast<uint16_t>(sections_.size()); 
    Elf64_Shdr symtab_shdr {.sh_type = SHT_SYMTAB, .sh_addralign=8, .sh_entsize = 0x18};
    vector<char> empty_vec;
    sections_.emplace_back(make_pair(symtab_shdr, empty_vec));
    section_names_.emplace_back(".symtab");
  }

  void CuElf::addNvInfo() {
    section_idx_[".nv.info"] = static_cast<uint16_t>(sections_.size()); 
    Elf64_Shdr nv_info_shdr {.sh_type = CUDA_INFO, .sh_addralign=4};
    vector<char> empty_vec;
    sections_.emplace_back(make_pair(nv_info_shdr, empty_vec));
    section_names_.emplace_back(".nv.info");
  }

  void CuElf::addNvInfoN(ElfKernel const* elf_kernel){
    string name = string(".nv.info.") + elf_kernel->name_;
    section_idx_[name] = static_cast<uint16_t>(sections_.size()); 
    Elf64_Shdr nv_info_n_shdr {.sh_type = CUDA_INFO, .sh_addralign=4}; 
    vector<char> info_data;
    sections_.emplace_back(make_pair(nv_info_n_shdr, info_data));
    section_names_.emplace_back(name);
  }

  void CuElf::addConst0(ElfKernel const* elf_kernel){
    string name = string(".nv.constant0.") + elf_kernel->name_;
    section_idx_[name] = static_cast<uint16_t>(sections_.size()); 
    Elf64_Shdr const_shdr {.sh_type = SHT_PROGBITS, .sh_flags=SHF_ALLOC, .sh_addralign=4};
    int const_size = elf_kernel->const_size_;
    vector<char> data (const_size, '\x00');
    sections_.emplace_back(make_pair(const_shdr, data));
    section_names_.emplace_back(name);
  }

  void CuElf::addText(ElfKernel const* elf_kernel){
    string name = string(".text.") + elf_kernel->name_;
    section_idx_[name] = static_cast<uint16_t>(sections_.size()); 

    int shdr_idx = sections_.size();
    text_section_idx_[elf_kernel->name_] = shdr_idx;

    // FIXME: .sh_info should be .text.X idx in symtab
    Elf64_Shdr text_shdr {.sh_type = SHT_PROGBITS, 
                          .sh_flags=6, .sh_info=4, .sh_addralign=128};
    text_shdr.sh_flags += elf_kernel->num_bars_ << 20;
    text_shdr.sh_info += elf_kernel->num_regs_ << 24;
    vector<char> data (elf_kernel->data_);
    sections_.emplace_back(make_pair(text_shdr, data));
    section_names_.emplace_back(name);
  }

  void CuElf::setShstrtab() {
    vector<char>& shstrtab_data = sections_[section_idx_.at(".shstrtab")].second;
    for(int i=0; i<sections_.size(); ++i){
      auto& [header, data] = sections_[i];
      header.sh_name = shstrtab_data.size();
      string& curr_str = section_names_[i];
      shstrtab_data.insert(shstrtab_data.end(), curr_str.begin(), curr_str.end());
      shstrtab_data.push_back('\x00');
    }
  }

  void CuElf::setSymtab() {
    // first is null
    symbols_.emplace_back();
    symbol_names_.emplace_back();
    uint32_t curr_st_name = 1;
    for(auto const& elf_kernel : elf_kernels_){
      // .text.name & .nv.constant0.name
      Elf64_Sym text_entry {.st_info = STT_SECTION};
      string text_name = string(".text.") + elf_kernel->name_;
      int idx = section_idx_.at(text_name);
      text_entry.st_shndx = idx;
      text_entry.st_name = curr_st_name;
      curr_st_name += text_name.size() + 1;
      symbols_.push_back(text_entry);
      symbol_names_.emplace_back(text_name);

      // .nv.constant0.name
      Elf64_Sym const0_entry {.st_info = STT_SECTION};
      string const0_name = string(".nv.constant0.") + elf_kernel->name_;
      idx = section_idx_.at(const0_name);
      const0_entry.st_shndx = idx;
      const0_entry.st_name = curr_st_name;
      curr_st_name += const0_name.size() + 1;
      symbols_.push_back(const0_entry);
      symbol_names_.emplace_back(const0_name);
    }

    // each kernel has its own symtab entry
    for(auto const& elf_kernel : elf_kernels_){
      Elf64_Sym kernel_entry {.st_info = 0x12, .st_other = 0x10};
      int idx = text_section_idx_.at(elf_kernel->name_);
      kernel_entry.st_shndx = idx;
      kernel_entry.st_size = elf_kernel->data_.size();
      // Record its idx
      symbol_func_idx_[elf_kernel->name_] = symbols_.size();
      // Record symtol entry
      kernel_entry.st_name = curr_st_name;
      curr_st_name += elf_kernel->name_.size() + 1;
      symbols_.push_back(kernel_entry);
      symbol_names_.emplace_back(elf_kernel->name_);
    }

    // write data to symtab
    int symtab_idx = section_idx_.at(".symtab");
    vector<char>& symtab_data = sections_[symtab_idx].second;
    for(Elf64_Sym & sym : symbols_){
      char* sym_ptr = reinterpret_cast<char*>(&sym);
      symtab_data.insert(symtab_data.end(), sym_ptr, sym_ptr + sizeof(sym));
    }

    Elf64_Shdr& symtab_hdr = sections_[symtab_idx].first;
    symtab_hdr.sh_link = section_idx_.at(".strtab");
    symtab_hdr.sh_info = section_idx_.at(".symtab");
    // write string to strtab
    uint32_t curr_str_offset = 0;
    vector<char>& strtab_data = sections_[section_idx_.at(".strtab")].second;
    for(int i=0; i<symbols_.size(); ++i){
      symbols_[i].st_name = curr_str_offset;
      string& curr_str = symbol_names_[i];
      strtab_data.insert(strtab_data.end(), curr_str.begin(), curr_str.end());
      strtab_data.push_back('\x00');
      curr_str_offset += curr_str.length() + 1;
    }
    Elf64_Shdr& strtab_hdr = sections_[section_idx_.at(".strtab")].first;
    strtab_hdr.sh_size = curr_str_offset;
  }

  void CuElf::setNvInfo() {
    struct NvInfoEntry {
      char e0 = 4; // encoding info 0
      char e1;
      uint16_t e2 = 8;
      uint32_t symtab_idx;
      uint32_t value = 0;
    };
    // For each kernel. We need to provide 
    // MAX_STACK_SIZE MIN_STACK_SIZE FRAME_SIZE
    auto& [header, data] = sections_[section_idx_.at(".nv.info")];
    // Set link to symtab
    header.sh_link = section_idx_.at(".symtab");
    for(auto const& elf_kernel : elf_kernels_){
      uint32_t symtab_idx = symbol_func_idx_.at(elf_kernel->name_);
      NvInfoEntry max_stack_size {.e1=0x23, .symtab_idx=symtab_idx};
      NvInfoEntry min_stack_size {.e1=0x12, .symtab_idx=symtab_idx};
      NvInfoEntry frame_size     {.e1=0x11, .symtab_idx=symtab_idx};
      data.insert(data.end(), reinterpret_cast<char*>(&max_stack_size),
                              reinterpret_cast<char*>(&max_stack_size)+sizeof(NvInfoEntry));
      data.insert(data.end(), reinterpret_cast<char*>(&min_stack_size),
                              reinterpret_cast<char*>(&min_stack_size)+sizeof(NvInfoEntry));
      data.insert(data.end(), reinterpret_cast<char*>(&frame_size),
                              reinterpret_cast<char*>(&frame_size)+sizeof(NvInfoEntry));
    }
  }

  void CuElf::setNvInfoN() {
    struct ParameterCbank {
      char e0 = 4;
      char e1 = 0xa;
      uint16_t e2 = 8;
      uint32_t ord; // ord of kernel (start from 1) * 2
      uint16_t param_base = 0x160;
      uint16_t param_size;
    };

    struct ParameterSize {
      char e0 = 3;
      char e1 = 0x19;
      uint16_t param_size;
    };

    struct ParameterInfo {
      ParameterInfo(uint16_t p_ord, uint16_t p_offset, uint32_t size)
      : ord(p_ord), offset(p_offset){
        flag = 0x1f000;
        flag |= (size/4) << 20;
      }
      char e0 = 4;
      char e1 = 0x17;
      uint16_t e2 = 0xc;
      uint32_t e3 = 0;
      uint16_t ord;
      uint16_t offset;
      uint32_t flag;
    };

    struct MaxRegCount {
      char e0 = 3;
      char e1 = 0x1b;
      uint16_t max_regs = 0xff;
    };

    struct ExitOffsets {
      char e0 = 4;
      char e1 = 0x1c;
      uint16_t num_exits;
      // uint32_t offset for each exit
    };

    uint32_t symtab_idx = section_idx_.at(".symtab");
    // Track ordinal of kernels, start with 2. (step = 2)
    uint32_t ordinal = 2; 

    for(auto const& elf_kernel : elf_kernels_){
      string name = string(".nv.info.") + elf_kernel->name_;
      auto& [header, data] = sections_[section_idx_.at(name)];
      header.sh_link = symtab_idx;
      // sh_info is .text.{name}'s section_idx
      header.sh_info = section_idx_.at(string(".text.") + elf_kernel->name_);

      // Add data
      // 1. parameter (base, list of parameters) 
      // EIATTR_PARAM_CBANK
      uint16_t param_size = reduce(elf_kernel->param_sizes_.begin(), elf_kernel->param_sizes_.end());
      ParameterCbank param_cbank {.ord=ordinal, .param_size=param_size};
      data.insert(data.end(), reinterpret_cast<char*>(&param_cbank),
                              reinterpret_cast<char*>(&param_cbank)+sizeof(param_cbank));
      // EIATTR_CBANK_PARAM_SIZE
      ParameterSize parameter_size {.param_size=param_size};
      data.insert(data.end(), reinterpret_cast<char*>(&parameter_size),
                              reinterpret_cast<char*>(&parameter_size)+sizeof(parameter_size));

      // EIATTR_KPARAM_INFO (for each parameter)
      vector<int> p_offset;
      exclusive_scan(elf_kernel->param_sizes_.begin(), elf_kernel->param_sizes_.end(), 
                     back_inserter(p_offset), 0);
      for(int i=elf_kernel->param_sizes_.size()-1; i>=0; --i){
        ParameterInfo param_info(i, p_offset[i], elf_kernel->param_sizes_[i]);
        data.insert(data.end(), reinterpret_cast<char*>(&param_info),
                                reinterpret_cast<char*>(&param_info)+sizeof(param_info));
      }

      // 2. max_reg_count 
      // EIATTR_MAXREG_COUNT
      MaxRegCount maxreg_count;
      data.insert(data.end(), reinterpret_cast<char*>(&maxreg_count),
                              reinterpret_cast<char*>(&maxreg_count)+sizeof(maxreg_count));
      
      // 3. exit offset 
      // EIATTR_EXIT_INSTR_OFFSETS
      ExitOffsets exit_offsets {.num_exits=static_cast<uint16_t>(elf_kernel->exit_offsets_.size()*4)};
      data.insert(data.end(), reinterpret_cast<char*>(&exit_offsets),
                              reinterpret_cast<char*>(&exit_offsets)+sizeof(exit_offsets));
      for(uint32_t exit_offset : elf_kernel->exit_offsets_){
        data.insert(data.end(), (char*)(&exit_offset), (char*)(&exit_offset)+sizeof(exit_offset));
      }

      ordinal += 2;
    } // for each elf_kernel
  }

  void CuElf::setConst0() {
    for(auto const& elf_kernel : elf_kernels_){
      string name = string(".nv.constant0.") + elf_kernel->name_;
      auto& [header, data] = sections_[section_idx_.at(name)];

      // sh_info is .text.{name}'s section_idx
      header.sh_info = section_idx_.at(string(".text.") + elf_kernel->name_);
    }
  }

  void CuElf::updateSizeOffset() {
    vector<uint32_t> section_sizes(sections_.size());
    // start with Ehdr
    uint64_t curr_offset = sizeof(header_);
    section_data_size_ = 0;
    for(int i=0; i<sections_.size(); ++i){
      auto& [header, data] = sections_[i];
      section_data_size_ += data.size();
      section_sizes[i] = data.size();
      header.sh_size = data.size();
      header.sh_offset = curr_offset;
      curr_offset += data.size();
    }
  }

  void CuElf::addProgramHeaders() {
    Elf64_Phdr p_hdr {.p_type=6, .p_flags=5, 
      .p_filesz=sizeof(p_hdr)*3, .p_memsz=sizeof(p_hdr)*3, .p_align=8}; // 3 program headers in total
    p_hdr.p_offset = sizeof(header_) + section_data_size_ + sizeof(Elf64_Shdr)*sections_.size();

    Elf64_Phdr p_progbits {.p_type=1, .p_flags=5, .p_align=8};
    // p_progbits.offset = first contains ... constant0
    uint64_t progbits_offset = numeric_limits<uint64_t>::max();
    uint64_t progbits_size = 0;

    for(int i=0; i<sections_.size(); ++i){
      string const& curr_str = section_names_[i];
      if(curr_str.find(".nv.constant0.") == 0 ||
         curr_str.find(".text.") == 0){
        auto const& [header, data] = sections_[i];
        progbits_size += header.sh_size;
        progbits_offset = min(progbits_offset, header.sh_offset);
      }
    }
    p_progbits.p_offset = progbits_offset;
    p_progbits.p_filesz = progbits_size;
    p_progbits.p_memsz  = progbits_size;

    Elf64_Phdr p_nobits {.p_type=1, .p_flags=6, .p_align=8};

    program_headers_.push_back(p_hdr);
    program_headers_.push_back(p_progbits);
    program_headers_.push_back(p_nobits);
  }

  void CuElf::setHeader() {
    // Header -->  Data  --> Section Headres --> Program Headers
    header_.e_shnum = sections_.size();
    header_.e_phnum = program_headers_.size();
    header_.e_shoff = sizeof(header_) + section_data_size_;
    header_.e_phoff = sizeof(header_) + section_data_size_ + sections_.size()*sizeof(Elf64_Shdr);
  }




} // namespace dada

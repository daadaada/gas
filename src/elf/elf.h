#pragma once

/* Discard support for elf32. */

/* 64-bit ELF base types. */
typedef uint64_t	Elf64_Addr;
typedef uint16_t	Elf64_Half;
typedef int64_t 	Elf64_SHalf;
typedef uint64_t	Elf64_Off;
typedef int32_t 	Elf64_Sword;
typedef uint32_t	Elf64_Word;
typedef uint64_t	Elf64_Xword;
typedef int64_t 	Elf64_Sxword;

/* These constants are for the segment types stored in the image headers */
#define PT_NULL    0
#define PT_LOAD    1
#define PT_DYNAMIC 2
#define PT_INTERP  3
#define PT_NOTE    4
#define PT_SHLIB   5
#define PT_PHDR    6
#define PT_TLS     7               /* Thread local storage segment */
#define PT_LOOS    0x60000000      /* OS-specific */
#define PT_HIOS    0x6fffffff      /* OS-specific */
#define PT_LOPROC  0x70000000
#define PT_HIPROC  0x7fffffff
#define PT_GNU_EH_FRAME		0x6474e550

#define PT_GNU_STACK	(PT_LOOS + 0x474e551)

/*
 * Extended Numbering
 *
 * If the real number of program header table entries is larger than
 * or equal to PN_XNUM(0xffff), it is set to sh_info field of the
 * section header at index 0, and PN_XNUM is set to e_phnum
 * field. Otherwise, the section header at index 0 is zero
 * initialized, if it exists.
 *
 * Specifications are available in:
 *
 * - Oracle: Linker and Libraries.
 *   Part No: 817–1984–19, August 2011.
 *   http://docs.oracle.com/cd/E18752_01/pdf/817-1984.pdf
 *
 * - System V ABI AMD64 Architecture Processor Supplement
 *   Draft Version 0.99.4,
 *   January 13, 2010.
 *   http://www.cs.washington.edu/education/courses/cse351/12wi/supp-docs/abi.pdf
 */
#define PN_XNUM 0xffff

/* These constants define the different elf file types */
#define ET_NONE   0
#define ET_REL    1
#define ET_EXEC   2
#define ET_DYN    3
#define ET_CORE   4
#define ET_LOPROC 0xff00
#define ET_HIPROC 0xffff

/* This info is needed when parsing the symbol table */
#define STB_LOCAL  0
#define STB_GLOBAL 1
#define STB_WEAK   2

#define STT_NOTYPE  0
#define STT_OBJECT  1
#define STT_FUNC    2
#define STT_SECTION 3
#define STT_FILE    4
#define STT_COMMON  5
#define STT_TLS     6

#define ELF_ST_BIND(x)		((x) >> 4)
#define ELF_ST_TYPE(x)		(((unsigned int) x) & 0xf)
#define ELF64_ST_BIND(x)	ELF_ST_BIND(x)
#define ELF64_ST_TYPE(x)	ELF_ST_TYPE(x)

struct Elf64_Dyn{
  Elf64_Sxword d_tag;		/* entry tag value */
  union {
    Elf64_Xword d_val;
    Elf64_Addr d_ptr;
  } d_un;
};

/* The following are used with relocations */
#define ELF64_R_SYM(i)			((i) >> 32)
#define ELF64_R_TYPE(i)			((i) & 0xffffffff)

struct Elf64_Rel {
  Elf64_Addr r_offset;	/* Location at which to apply the action */
  Elf64_Xword r_info;	/* index and type of relocation */
};

struct Elf64_Rela {
  Elf64_Addr r_offset;	/* Location at which to apply the action */
  Elf64_Xword r_info;	/* index and type of relocation */
  Elf64_Sxword r_addend;	/* Constant addend used to compute value */
} ;

struct Elf64_Sym {
  Elf64_Word st_name = 0;		/* Symbol name, index in string tbl */
  unsigned char	st_info = 0;	/* Type and binding attributes */
  unsigned char	st_other = 0;	/* No defined meaning, 0 */
  Elf64_Half st_shndx = 0;		/* Associated section index */
  Elf64_Addr st_value = 0;		/* Value of the symbol */
  Elf64_Xword st_size = 0;		/* Associated symbol size */
};


#define EI_NIDENT	16

struct Elf64_Ehdr {
  unsigned char	e_ident[EI_NIDENT];	/* ELF "magic number" */
  Elf64_Half e_type = 2;                /* 2: EXEC */
  Elf64_Half e_machine = 190;           /* 190: CUDA */
  Elf64_Word e_version = 101;           /* CUDA version. e.g., 101 for CUDA 10.1 */
  Elf64_Addr e_entry = 0;		/* Entry point virtual address */
  Elf64_Off e_phoff = 0;		/* Program header table file offset */
  Elf64_Off e_shoff = 0;		/* Section header table file offset */
  Elf64_Word e_flags = 0x500;           /* <arch>05<arch>. e.g., 0x460546 for volta (70) */
  Elf64_Half e_ehsize = 64;             /* 64: size of header */
  Elf64_Half e_phentsize = 56;          /* program header entry size */
  Elf64_Half e_phnum = 0;
  Elf64_Half e_shentsize = 64;          /* section header entry size */
  Elf64_Half e_shnum = 0;
  Elf64_Half e_shstrndx;
};

/* These constants define the permissions on sections in the program
   header, p_flags. */
#define PF_R		0x4
#define PF_W		0x2
#define PF_X		0x1

struct Elf64_Phdr {
  Elf64_Word p_type;
  Elf64_Word p_flags;
  Elf64_Off p_offset;		/* Segment file offset */
  Elf64_Addr p_vaddr;		/* Segment virtual address */
  Elf64_Addr p_paddr;		/* Segment physical address */
  Elf64_Xword p_filesz;		/* Segment size in file */
  Elf64_Xword p_memsz;		/* Segment size in memory */
  Elf64_Xword p_align;		/* Segment alignment, file & memory */
};

/* sh_type */
#define SHT_NULL	0
#define SHT_PROGBITS	1
#define SHT_SYMTAB	2
#define SHT_STRTAB	3
#define SHT_RELA	4
#define SHT_HASH	5
#define SHT_DYNAMIC	6
#define SHT_NOTE	7
#define SHT_NOBITS	8
#define SHT_REL		9
#define SHT_SHLIB	10
#define SHT_DYNSYM	11
#define SHT_NUM		12
#define SHT_LOPROC	0x70000000
#define CUDA_INFO   0x70000000
#define SHT_HIPROC	0x7fffffff
#define SHT_LOUSER	0x80000000
#define SHT_HIUSER	0xffffffff

/* sh_flags */
#define SHF_WRITE		0x1
#define SHF_ALLOC		0x2
#define SHF_EXECINSTR		0x4
#define SHF_RELA_LIVEPATCH	0x00100000
#define SHF_RO_AFTER_INIT	0x00200000
#define SHF_MASKPROC		0xf0000000

/* special section indexes */
#define SHN_UNDEF	0
#define SHN_LORESERVE	0xff00
#define SHN_LOPROC	0xff00
#define SHN_HIPROC	0xff1f
#define SHN_LIVEPATCH	0xff20
#define SHN_ABS		0xfff1
#define SHN_COMMON	0xfff2
#define SHN_HIRESERVE	0xffff

struct Elf64_Shdr {
  Elf64_Word sh_name = 0;		/* Section name, index in string tbl */
  Elf64_Word sh_type = 0;		/* Type of section */
  Elf64_Xword sh_flags = 0;		/* Miscellaneous section attributes */
  Elf64_Addr sh_addr = 0;		/* Section virtual addr at execution */
  Elf64_Off sh_offset = 0;		/* Section file offset */
  Elf64_Xword sh_size = 0;		/* Size of section in bytes */
  Elf64_Word sh_link = 0;		/* Index of another section */
  Elf64_Word sh_info = 0;		/* Additional section information */
  Elf64_Xword sh_addralign = 0;	/* Section alignment */
  Elf64_Xword sh_entsize = 0;	/* Entry size if section holds table */
};

#define	EI_MAG0		0		/* e_ident[] indexes */
#define	EI_MAG1		1
#define	EI_MAG2		2
#define	EI_MAG3		3
#define	EI_CLASS	4
#define	EI_DATA		5
#define	EI_VERSION	6
#define	EI_OSABI	7
#define	EI_PAD		8

#define	ELFMAG0		0x7f		/* EI_MAG */
#define	ELFMAG1		'E'
#define	ELFMAG2		'L'
#define	ELFMAG3		'F'
#define	ELFMAG		"\177ELF"
#define	SELFMAG		4

#define	ELFCLASSNONE	0		/* EI_CLASS */
#define	ELFCLASS32	1
#define	ELFCLASS64	2
#define	ELFCLASSNUM	3

#define ELFDATANONE	0		/* e_ident[EI_DATA] */
#define ELFDATA2LSB	1
#define ELFDATA2MSB	2

#define EV_NONE		0		/* e_version, EI_VERSION */
#define EV_CURRENT	1
#define EV_NUM		2

#define ELFOSABI_NONE	0
#define ELFOSABI_LINUX	3

#ifndef ELF_OSABI
#define ELF_OSABI ELFOSABI_NONE
#endif

/*
 * Notes used in ET_CORE. Architectures export some of the arch register sets
 * using the corresponding note types via the PTRACE_GETREGSET and
 * PTRACE_SETREGSET requests.
 */
#define NT_PRSTATUS	1
#define NT_PRFPREG	2
#define NT_PRPSINFO	3
#define NT_TASKSTRUCT	4
#define NT_AUXV		6

/* Note header in a PT_NOTE section */
struct Elf64_Nhdr {
  Elf64_Word n_namesz;	/* Name size */
  Elf64_Word n_descsz;	/* Content size */
  Elf64_Word n_type;	/* Content type */
};

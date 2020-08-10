#pragma once

#include "core/module.h"
#include "core/kernel.h"
#include "core/instruction.h"
#include "core/operand.h"

#include "algorithms/cfg.h"
#include "algorithms/register_allocator.h"
#include "algorithms/stall_setter.h"
#include "algorithms/barrier_setter.h"

#include "elf/cu_elf.h"
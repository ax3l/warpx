#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This python script reads the orbital energies of neutral atoms in
atomic_orbital_data.txt (from the NIST website) and writes the binding energy
and occupation of each subshell to the C++ file AtomicOrbitalEnergiesTable.H.
"""

import os
import re

HARTREE_TO_EV = 27.211386245988  # CODATA 2018

elements = []
with open(os.path.join(".", "atomic_orbital_data.txt")) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        atomic_number, symbol, *orbitals = line.split()
        subshells = []
        for orbital in orbitals:
            match = re.fullmatch(r"\d[spdf]:(\d+):(-\d+\.\d+)", orbital)
            subshells.append(
                (-float(match.group(2)) * HARTREE_TO_EV, int(match.group(1)))
            )
        # From the most to the least tightly bound subshell
        subshells.sort(reverse=True)
        elements.append((int(atomic_number), symbol, subshells))

# The elements are complete, in order, and start at hydrogen
atomic_numbers = [element[0] for element in elements]
assert atomic_numbers == list(range(1, len(atomic_numbers) + 1))
for atomic_number, symbol, subshells in elements:
    assert sum(occupation for _, occupation in subshells) == atomic_number, symbol

offsets = [0]
for _, _, subshells in elements:
    offsets.append(offsets[-1] + len(subshells))

cpp_string = "// This script was automatically generated!\n"
cpp_string += (
    "// Edit dev/Source/Utils/Physics/write_atomic_orbital_data_cpp.py instead!\n\n"
)
cpp_string += "#ifndef WARPX_UTILS_PHYSICS_ATOMIC_ORBITAL_ENERGIES_TABLE_H_\n"
cpp_string += "#define WARPX_UTILS_PHYSICS_ATOMIC_ORBITAL_ENERGIES_TABLE_H_\n\n"
cpp_string += "namespace utils::physics\n"
cpp_string += "{\n"
cpp_string += (
    "    //! Elements with orbital energies are the ones up to this atomic number.\n"
)
cpp_string += "    constexpr int orbital_energies_max_atomic_number = "
cpp_string += str(atomic_numbers[-1]) + ";\n\n"

cpp_string += (
    "    //! First subshell of the element with atomic number Z is at index Z-1.\n"
)
cpp_string += "    constexpr int orbital_subshell_offsets[orbital_energies_max_atomic_number + 1]{"
for count, offset in enumerate(offsets):
    cpp_string += "\n        " if count % 15 == 0 else " "
    cpp_string += str(offset) + ","
cpp_string = cpp_string[:-1] + "\n    };\n\n"

cpp_string += "    /** Binding energies [eV] of the subshells of neutral atoms.\n"
cpp_string += "     *\n"
cpp_string += (
    "     * These go from the most to the least bound subshell of each element.\n"
)
cpp_string += "     * Source and notes: atomic_orbital_data.txt\n"
cpp_string += "     */\n"
cpp_string += (
    "    constexpr double table_orbital_binding_energies[" + str(offsets[-1]) + "]{"
)
for _, symbol, subshells in elements:
    cpp_string += "\n        // " + symbol
    for count, (energy, _) in enumerate(subshells):
        cpp_string += "\n        " if count % 5 == 0 else " "
        cpp_string += f"{energy:.6f},"
cpp_string = cpp_string[:-1] + "\n    };\n\n"

cpp_string += (
    "    //! Number of electrons in the subshells of table_orbital_binding_energies.\n"
)
cpp_string += "    constexpr int table_orbital_occupations[" + str(offsets[-1]) + "]{"
count = 0
for _, _, subshells in elements:
    for _, occupation in subshells:
        cpp_string += "\n        " if count % 24 == 0 else " "
        cpp_string += str(occupation) + ","
        count += 1
cpp_string = cpp_string[:-1] + "\n    };\n"
cpp_string += "}\n\n"
cpp_string += "#endif // #ifndef WARPX_UTILS_PHYSICS_ATOMIC_ORBITAL_ENERGIES_TABLE_H_\n"

with open(os.path.join(".", "AtomicOrbitalEnergiesTable.H"), "w") as f:
    f.write(cpp_string)

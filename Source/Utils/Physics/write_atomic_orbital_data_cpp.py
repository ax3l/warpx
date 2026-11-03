#!/usr/bin/env python3
#
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""
This python script reads the binding energies and occupations of the subshells of
neutral atoms in atomic_orbital_data.txt (from the Evaluated Atomic Data Library) and
writes them to the C++ file AtomicOrbitalEnergiesTable.H, grouped by electrons of the same
binding energy, from the most to the least bound electron.
"""

import os
import re

elements = []
with open(os.path.join(".", "atomic_orbital_data.txt")) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        atomic_number, symbol, *orbitals = line.split()
        atomic_number = int(atomic_number)
        shells = []
        for orbital in orbitals:
            match = re.fullmatch(r"\w+:([\d.]+):([\d.]+)", orbital)
            shells.append((float(match.group(2)), float(match.group(1))))
        # From the most to the least tightly bound subshell
        shells.sort(reverse=True)
        total_occupation = sum(occupation for _, occupation in shells)
        assert abs(total_occupation - atomic_number) < 1.0e-3, symbol

        # The occupations of partially filled subshells can be fractional, while each electron
        # of an ion needs one binding energy: use the mean binding energy over the share of the
        # subshells of each electron, which keeps the total binding energy of the atom.
        binding_of_electron = []
        for electron in range(atomic_number):
            lower = 0.0
            mean = 0.0
            for energy, occupation in shells:
                upper = lower + occupation
                mean += energy * max(
                    0.0, min(upper, electron + 1) - max(lower, electron)
                )
                lower = upper
            # The occupations of the source add up to the atomic number within rounding.
            mean += shells[-1][0] * max(0.0, electron + 1 - max(lower, electron))
            binding_of_electron.append(mean)

        subshells = []
        for energy in binding_of_electron:
            if subshells and abs(subshells[-1][0] - energy) < 1.0e-6:
                subshells[-1][1] += 1
            else:
                subshells.append([energy, 1])
        elements.append((atomic_number, symbol, subshells))

# The elements are complete, in order, and start at hydrogen
atomic_numbers = [element[0] for element in elements]
assert atomic_numbers == list(range(1, len(atomic_numbers) + 1))

offsets = [0]
for _, _, subshells in elements:
    offsets.append(offsets[-1] + len(subshells))

cpp_string = "// This script was automatically generated!\n"
cpp_string += (
    "// Edit dev/Source/Utils/Physics/write_atomic_orbital_data_cpp.py instead!\n\n"
)
cpp_string += (
    "// The data in this file is licensed under CC BY 4.0, unlike the code of WarpX, and\n"
    "// comes without warranties: https://creativecommons.org/licenses/by/4.0/\n"
    "// Adapted from the Evaluated Atomic Data Library (EADL), by Dermott E. Cullen, from\n"
    "// EPICS2025, Lawrence Livermore National Laboratory, https://nuclear.llnl.gov/EPICS/\n"
    "// The source and the changes to the data are in atomic_orbital_data.txt. In addition, this\n"
    "// file sorts the subshells and assigns one binding energy to each electron.\n\n"
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

cpp_string += "    //! First group of electrons of the element with atomic number Z is at index Z-1.\n"
cpp_string += "    constexpr int orbital_subshell_offsets[orbital_energies_max_atomic_number + 1]{"
for count, offset in enumerate(offsets):
    cpp_string += "\n        " if count % 15 == 0 else " "
    cpp_string += str(offset) + ","
cpp_string = cpp_string[:-1] + "\n    };\n\n"

cpp_string += "    /** Binding energies [eV] of groups of electrons of neutral atoms.\n"
cpp_string += "     *\n"
cpp_string += (
    "     * These go from the most to the least bound electrons of each element.\n"
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
    "    //! Number of electrons in the groups of table_orbital_binding_energies.\n"
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

"""
This file is part of VDISCOVER.

VDISCOVER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

VDISCOVER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with VDISCOVER. If not, see <http://www.gnu.org/licenses/>.

Copyright 2014 by G.Grieco
"""

from math import ceil

from Types import Type
from ptrace.error import PtraceError


def FindModule(value, mm):
    return mm.findModule(value)


def RefinePType(ptype, value, process, mm):

    if value is None:
        return (Type("Top32", 4), value)

    if str(ptype) == "Ptr32":
        ptr = value
        if ptr == 0x0:
            return (Type("NPtr32", 4), ptr)
        else:

            try:
                _ = process.readBytes(ptr, 4)
            except PtraceError:
                return (Type("DPtr32", 4), ptr)

            mm.checkPtr(ptr)
            if mm.isStackPtr(ptr):
                return (Type("SPtr32", 4), ptr)
            elif mm.isHeapPtr(ptr):
                return (Type("HPtr32", 4), ptr)
            elif mm.isCodePtr(ptr):
                return (Type("GxPtr32", 4), ptr)
            elif mm.isFilePtr(ptr):
                return (Type("FPtr32", 4), ptr)
            elif mm.isGlobalPtr(ptr):
                return (Type("GPtr32", 4), ptr)
            else:
                return (Type("Ptr32", 4), ptr)

    elif str(ptype) == "Num32":
        num = value
        if num == 0x0:
            return (Type("Num32B0", 4), num)
        else:
            binlen = len(bin(num)) - 2
            binlen = int(ceil(binlen / 8.0)) * 8
            return (Type("Num32B" + str(binlen), 4), num)

    return (Type("Top32", 4), value)

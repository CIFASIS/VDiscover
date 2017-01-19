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

import copy


class Type:

    def __init__(self, name, size, index=None):
        self.name = str(name)
        self.size_in_bytes = size
        self.index = index

    def __str__(self):

        r = str(self.name)
        if (self.index is not None):
            r = r + "(" + str(self.index) + ")"

        return r

    def getSize(self):
        return self.size_in_bytes

    # def copy(self):
    #  return copy.copy(self)

ptypes = [Type("Num32", 4, None),
          Type("Ptr32", 4, None),  # Generic pointer
          Type("SPtr32", 4, None),  # Stack pointer
          Type("HPtr32", 4, None),  # Heap pointer
          Type("GxPtr32", 4, None),  # Global eXecutable pointer
          Type("FPtr32", 4, None),  # File pointer
          Type("NPtr32", 4, None),  # NULL pointer
          Type("DPtr32", 4, None),  # Dangling pointer
          Type("GPtr32", 4, None),  # Global pointer
          Type("Top32", 4, None)
          ]

for i in range(0, 33, 8):
    ptypes.append(Type("Num32B" + str(i), 4, None))

num32_ptypes = filter(lambda t: "Num32" in str(t), ptypes)
ptr32_ptypes = ptypes[1:9]
generic_ptypes = [Type("Top32", 4, None)]


def isNum(ptype):
    return ptype in ["int", "ulong", "long", "char"]


def isPtr(ptype):
    return "addr" in ptype or "*" in ptype or "string" in ptype or "format" in ptype or "file" in ptype


def isVoid(ptype):
    return ptype == "void"


def isNull(val):
    return val == "0x0" or val == "0"


def GetPtype(ptype):

    if isPtr(ptype):
        return Type("Ptr32", 4)
    elif isNum(ptype):
        return Type("Num32", 4)
    elif isVoid(ptype):
        return Type("Top32", 4)
    else:
        return Type("Top32", 4)

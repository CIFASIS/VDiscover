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


def prepare_inputs(inputs):
    r = []
    for input in inputs:
        arg = input.PrepareData()
        if not (arg is None):
            r.append(arg)

    return r


class Input:

    data = None
    concrete = False

    def __init__(self):
        pass

    def __len__(self):
        return len(self.data)

    def copy(self):
        # print "data:",self.data
        return copy.copy(self)

    def isSymbolic(self):
        return not self.concrete

    def isConcrete(self):
        return self.concrete

    def SetSymbolic(self):
        self.concrete = False

    def SetConcrete(self):
        self.concrete = True


class Arg(Input):

    def __init__(self, i, data):
        self.i = i

        self.data = str(data)
        if ("\0" in data):
            self.data = self.data.split("\0")[0]

        self.size = len(self.data)

    def __str__(self):
        return "Arg(" + str(self.i) + ") = " + repr(self.data)

    def GetData(self):
        return str(self.data)

    def GetSize(self):
        return len(self.data)

    def PrepareData(self):

        return self.GetData()

    def IsValid(self):
        return self.size > 0

    def __cmp__(self, arg):
        return cmp(self.i, arg.i)

    def GetName(self):
        if self.concrete:
            return "cargv_" + str(self.i)
        else:
            return "argv_" + str(self.i)

    def GetType(self):
        return "arg"


# class Env(Input):
#   def __init__(self, name, data):
#     self.name = name
#
#     self.data = str(data)
#     if ("\0" in data):
#       self.data = self.data.split("\0")[0]
#
#     self.size = len(self.data)
#
#   def GetData(self):
#     return str(self.data)
#
#   def GetSize(self):
#     return len(self.data)
#
#   def PrepareData(self):
#
#     return self.GetData()
#
#   def IsValid(self):
#     return self.size > 0
#
#   def __cmp__(self, arg):
#     return cmp(self.i, arg.i)
#
#   def copy(self):
#     return Arg(self.i, self.data)
#
#   def GetName(self):
#     return "env_"+str(self.i)
#
#   def GetType(self):
#     return "env"

class File(Input):

    def __init__(self, filename, data):
        self.filename = str(filename)
        self.data = str(data)
        self.size = len(data)

    def __str__(self):
        return "file(" + str(self.filename) + ") = " + repr(self.data)

    def GetData(self):
        return str(self.data)

    def GetSize(self):
        return len(self.data)

    def PrepareData(self):
        if self.filename == "/dev/stdin":
            with open("Stdin", 'w') as f:
                f.write(self.data)

            return "< Stdin"
        else:
            with open(self.filename, 'w') as f:
                f.write(self.data)

            return None

    def IsValid(self):
        return True

#  def copy(self):
#    return File(self.filename, self.data)

    def GetName(self):
        return "file_" + self.filename.replace("/", "__")

    def GetFilename(self):
        return str(self.filename)

    def GetType(self):
        return "file"

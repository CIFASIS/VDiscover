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

import os
import os.path
import shutil

from Input import Arg, File


def GetCmd(s):

    if os.path.exists("path.txt"):
        f = open("path.txt")
        x = f.readline()
        return x.replace("\n", "").strip(" ")
    else:
        return s


def GetArg(n, conc):

    if conc:
        filename = "cargv_" + str(n) + ".symb"
        data = open(filename).read()
        x = Arg(n, data)
        x.SetConcrete()
    else:
        filename = "argv_" + str(n) + ".symb"
        data = open(filename).read()
        x = Arg(n, data)
        x.SetSymbolic()

    return x


def WriteTestcase(name, program, args, copy=False):
    try:
        os.mkdir(name)
    except:
        pass

    os.chdir(name)
    filename = "path.txt"
    open(filename, "w").write(program)

    try:
        os.mkdir("inputs")
    except:
        pass

    os.chdir("inputs")
    for i, arg in enumerate(args):
        if "file:" in arg:
            # print arg
            arg = arg.replace("file:", "")
            assert(arg[0] == '/')
            filename = os.path.split(arg)[-1]
            # print filename
            if copy:
                shutil.copyfile(os.path.realpath(arg), "file_" + filename)
            else:
                os.symlink(os.path.realpath(arg), "file_" + filename)
            arg = filename

        filename = "argv_" + str(i + 1) + ".symb"
        open(filename, "w").write(arg)

    os.chdir("../..")


def GetArgs():
    #i = 1
    r = []

    for _, _, files in os.walk('.'):
        for f in files:
            # print f
            for i in range(10):
                # print str(i), f

                if ("cargv_" + str(i)) in f:
                    x = GetArg(i, True)
                    if x.IsValid():
                        r.append(x)

                    break

                elif ("argv_" + str(i)) in f:
                    x = GetArg(i, False)
                    if x.IsValid():
                        r.append(x)

                    break

    r.sort()
    # print r
    for i in range(len(r)):
        if r[i].i != i + 1:
            r = r[0:i]
            break

    # print r
    return r


def GetFile(filename, source):
    #size = int(os.path.getsize(source))
    data = open(source).read()
    return File(filename, data)


def GetFiles():

    r = []
    stdinf = "file___dev__stdin.symb"

    for dir, _, files in os.walk('.'):
        if dir == '.':
            for f in files:
                if (stdinf == f):
                    r.append(GetFile("/dev/stdin", stdinf))
                elif ("file_" in f):
                    filename = f.split(".symb")[0]
                    #filename = f.replace(".symb","")
                    filename = filename.split("file_")[1]
                    filename = filename.replace(".__", "")
                    x = GetFile(filename, f)
                    if x.IsValid():
                        r.append(x)

    return r

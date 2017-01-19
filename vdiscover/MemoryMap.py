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


class MemoryMaps:

    def __init__(self, path, pid):
        self.path = str(path)
        self.pid = pid
        self.update()

    def update(self):

        self.mm = dict()
        self.atts = dict()

        for line in open('/proc/' + str(self.pid) + '/maps'):
            line = line.replace("\n", "")
            # print line
            x = line.split(" ")

            mrange = x[0].split("-")
            mrange = map(lambda s: int(s, 16), mrange)
            # print tuple(mrange)

            self.mm[tuple(mrange)] = x[-1]
            self.atts[tuple(mrange)] = x[1]

    def isStackPtr(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return zone == "[stack]"
        return False

    def isHeapPtr(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return zone == "[heap]"
        return False

    def isCodePtr(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[
                    1] and 'x' in self.atts[mrange]:
                return True
        return False

    def isLibPtr(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return "/lib/" in zone
        return False

    def isGlobalPtr(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return zone == self.path
        return False

    def isFilePtr(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return zone == ""
        return False

    def checkPtr(self, ptr, update=True):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return True

        if update:
            self.update()
        else:
            return False

        return self.checkPtr(ptr, update=False)

    def findModule(self, ptr):
        for (mrange, zone) in self.mm.items():
            if ptr >= mrange[0] and ptr < mrange[1]:
                return str(zone)
        return None

    def __str__(self):
        r = ""
        for (mrange, zone) in self.mm.items():
            r = r + hex(mrange[0]) + " - " + \
                hex(mrange[1]) + " -> " + zone + "\n"
        return r

    def items(self):
        r = []
        for (x, y) in self.mm.items():
            r.append((x, y, self.atts[x]))

        return r

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

import sys
import csv
import copy

from Event    import Call, Crash, Abort, Exit, Timeout, Signal, Vulnerability, specs
from Types    import ptypes, isPtr, isNum, ptr32_ptypes, num32_ptypes, generic_ptypes

class TypePrinter:
  def __init__(self, filename, pname, mclass):
    self.tests = set()
    self.outfile = open(filename, "a+")
    self.pname = pname
    self.mclass = mclass
    self.csvwriter = csv.writer(self.outfile, delimiter='\t')

  def preprocess(self, event):

    r = list()

    if isinstance(event, Call):
      (name, args) = event.GetTypedName()

      for (index, arg) in enumerate(args[:]):
        r.append((name+":"+str(index),str(arg)))

    elif isinstance(event, Abort):
      (name, fields) = event.GetTypedName()
      r.append((name+":eip",str(fields[0])))

    elif isinstance(event, Exit):
      (name, fields) = event.GetTypedName()
      r.append((name,str(())))

    elif isinstance(event, Crash):
      (name, fields) = event.GetTypedName()
      r.append((name+":eip",str(fields[0])))

    elif isinstance(event, Vulnerability):
      (name, fields) = event.GetTypedName()
      r.append((name,str(fields[0])))

    elif isinstance(event, Timeout):
      (name, fields) = event.GetTypedName()
      r.append((name,str(())))

    elif isinstance(event, Signal):
      (name, fields) = event.GetTypedName()

      if name == "SIGSEGV":
        r.append((name+":addr",str(fields[0])))
      else:
        r.append((name,str(fields[0])))

    return r

  def print_events(self, label, events):

    r = list()

    for event in events:
      r = r + list(self.preprocess(event))

    events = r

    #x = hash(tuple(events))

    #if (x in self.tests):
    #  return

    #self.tests.add(x)

    trace = ""

    for x,y in events:
      trace = trace + ("%s=%s " % (x,y))

    row = [self.pname+":"+label,trace]

    if self.mclass is not None:
      row.append(self.mclass)

    self.csvwriter.writerow(row)
    self.outfile.flush()
    return row

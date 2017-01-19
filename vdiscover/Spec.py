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

realpath = os.path.dirname(os.path.realpath(__file__))
datadir = "data/"
f = open(realpath + "/" + datadir + "prototypes.conf")
specs = dict()

for raw_spec in f.readlines():
    raw_spec = raw_spec.replace("\n", "")
    raw_spec = raw_spec.replace(", ", ",")
    raw_spec = raw_spec.replace(" (", "(")
    raw_spec = raw_spec.replace("  ", " ")
    raw_spec = raw_spec.replace("  ", " ")
    if raw_spec != "" and raw_spec[0] != ";" and (not "SYS_" in raw_spec):
        x = raw_spec.split(" ")
        ret = x[0]
        x = x[1].split("(")
        name = x[0]
        param_types = x[1].replace(");", "").split(",")
        specs[name] = [ret] + param_types

# print specs

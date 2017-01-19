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

from Event import Call, Crash, Abort, Exit, Signal, Vulnerability
from Analysis import FindModule


def detect_vulnerabilities(preevents, events, process, mm):

    r = []

    for (i, event) in enumerate(events):
        r.append(detect_vulnerability(preevents, event, process, mm))

    return filter(lambda e: e is not None, r)


def detect_vulnerability(preevents, event, process, mm):

    if isinstance(event, Call):

        (name, args) = event.GetTypedName()
        if name == "system" or name == "popen":
            pass

    elif isinstance(event, Abort):

        if len(event.bt) > 0 and len(preevents) > 0:

            if not (str(preevents[-1]) in ["free", "malloc", "realloc"]):
                return None

            for (typ, val) in event.bt:
                module = FindModule(val, mm)
                if module == "[vdso]":
                    pass
                elif "libc-" in module:
                    assert(0)
                    return Vulnerability("MemoryCorruption")
                else:
                    return None

    elif isinstance(event, Crash):

        if str(
                event.fp_type[0]) == "DPtr32" and str(
                event.eip_type[0]) == "DPtr32":
            return Vulnerability("StackCorruption")

        for (typ, val) in event.bt:
            if str(typ) == "DPtr32":
                return Vulnerability("StackCorruption")

    elif isinstance(event, Signal):
        pass

    return None

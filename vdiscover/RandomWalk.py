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

import random
import time
import sys
import csv
import re

from ELF import ELF
from Spec import specs
from Misc import readmodfile


def RandomWalkElf(
        program,
        outfile,
        mclass,
        max_subtraces,
        max_explored_subtraces,
        min_size):

    csvwriter = csv.writer(open(outfile, "a+"), delimiter='\t')
    elf = ELF(program)

    # plt is inverted
    inv_plt = dict()

    for func, addr in elf.plt.items():
        if func in specs:  # external functions are discarded
            inv_plt[addr] = func

    elf.plt = inv_plt

    cond_control_flow_ins = ["jo", "jno", "js", "jns", "je",
                             "jz", "jnz", "jb", "jnae", "jc",
                             "jnb", "jae", "jnc", "jbe", "jna",
                             "ja", "jnbe", "jl", "jnge", "jge",
                             "jnl", "jle", "jng", "jg", "jnle",
                             "jp", "jpe", "jnp", "jpo", "jcxz", "jecxz"]

    ncond_control_flow_ins = ["ret", "jmp", "call", "retq", "jmp", "callq"]

    control_flow_ins = cond_control_flow_ins + ncond_control_flow_ins

    raw_inss = elf.GetRawInss()
    useful_inss_list = []
    useful_inss_dict = dict()
    libc_calls = []
    labels = dict()

    # print sys.argv[1]+"\t",
    #rclass = str(1)

    for i, ins in enumerate(raw_inss.split("\n")):

        # prefix removal
        ins = ins.replace("repz ", "")
        ins = ins.replace("rep ", "")

        pins = ins.split("\t")
        # print pins
        ins_addr = pins[0].replace(":", "").replace(" ", "")
        # print pins,ins_addr

        if len(pins) == 1 and ">" in ins:  # label
            # print ins
            # assert(0)
            x = pins[0].split(" ")

            ins_addr = x[0]

            y = [i, ins_addr, None, None]
            useful_inss_dict[ins_addr] = y
            useful_inss_list.append(y)

            # print "label:",y

        elif any(map(lambda x: x in ins, control_flow_ins)) and len(pins) == 3:  # control flow instruction
            # print pins
            x = pins[2].split(" ")

            ins_nme = x[0]
            ins_jaddr = x[-2]

            # if ("" == ins_jaddr):
            #  print pins
            # print x
            # print ins_nme, ins_jaddr
            y = [i, ins_addr, ins_nme, ins_jaddr]

            useful_inss_dict[ins_addr] = y
            useful_inss_list.append(y)

            if "call" in pins[2]:
                if ins_jaddr != '':
                    func_addr = int(ins_jaddr, 16)
                    if func_addr in elf.plt:
                        libc_calls.append(i)

        else:  # all other instructions
            y = [i, ins_addr, None, None]

            useful_inss_dict[ins_addr] = y
            useful_inss_list.append(y)

    # print useful_inss_list
    max_inss = len(useful_inss_list)
    traces = set()
    collected_traces = ""

    # exploration time!
    for _ in range(max_explored_subtraces):

        # resuling (sub)trace
        r = ""
        # starting point
        i = random.choice(libc_calls)
        j = 0

        #r = elf.path+"\t"
        r = ""

        while True:

            # last instruction case
            if (i + j) == max_inss:
                break

            _, ins_addr, ins_nme, ins_jaddr = useful_inss_list[i + j]

            # print i+j,ins_nme, ins_jaddr

            if ins_nme in ['call', 'callq']:  # ordinary call
                #"addr", ins_jaddr

                if ins_jaddr == '':
                    break  # parametric jmp, similar to ret for us

                ins_jaddr = int(ins_jaddr, 16)
                if ins_jaddr in elf.plt:
                    r = r + " " + elf.plt[ins_jaddr]
                    if elf.plt[ins_jaddr] == "exit":
                        break
                else:

                    if ins_jaddr in useful_inss_dict:
                        # assert(0)
                        #r = r + " " + hex(ins_jaddr)
                        i, _, _, _ = useful_inss_dict[ins_jaddr]
                        j = 0
                        continue

                    else:
                        pass  # ignored call

            elif ins_nme in ['ret', 'retq']:
                break
            else:
                pass
                # print i+j,ins_nme, ins_jaddr

            # print j
            if ins_nme == 'jmp':

                if ins_jaddr in elf.plt:  # call equivalent using jmp
                    r = r + " " + elf.plt[jaddr]

                else:

                    if ins_jaddr == '':
                        break  # parametric jmp, similar to ret for us

                    ins_jaddr = int(ins_jaddr, 16)
                    if ins_jaddr in useful_inss_dict:
                        #r = r + " " + hex(ins_jaddr)
                        i, _, _, _ = useful_inss_dict[ins_jaddr]
                        j = 0
                        continue

                    else:
                        pass  # ignored call

            if ins_nme in cond_control_flow_ins:

                assert(ins_jaddr is not None)

                cond = random.randint(0, 1)

                if cond == 1:

                    i, _, _, _ = useful_inss_dict[ins_jaddr]
                    j = 0
                    continue

            j = j + 1

        #r = r + "\t"+rclass
        x = hash(r)
        size = len(r.split(" ")) - 1

        # if x not in traces and size >= min_size:
        # print r+" .",
        collected_traces = collected_traces + r + " ."
        # traces.add(x)
        # if len(traces) >= max_subtraces:
        #  break

    row = [elf.path, collected_traces]
    if mclass is not None:
        row.append(mclass)

    csvwriter.writerow(row)

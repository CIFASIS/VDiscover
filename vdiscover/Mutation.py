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
import copy

from subprocess import Popen, PIPE, STDOUT

import Input


def opened_files(program, args, files, timeout=5):

    # check if the testcase is opened
    output = Popen(["timeout",
                    "-k",
                    "1",
                    str(timeout),
                    "strace",
                    "-e",
                    "open",
                    program] + args,
                   stdout=PIPE,
                   stderr=PIPE,
                   stdin=PIPE,
                   env=dict()).communicate()

    for mfile in files:
        filename = mfile.filename
        # print "checking",filename
        if 'open("' + filename in output[1]:
            return True

    return False
    # print output


def fuzz_cmd(prepared_inputs, fuzzer_cmd, seed):
    p = Popen(fuzzer_cmd.split(" ") + [str(seed)],
              stdout=PIPE, stdin=PIPE, stderr=PIPE)
    mutated_input = p.communicate(input=prepared_inputs)[0]
    return mutated_input.replace("\0", "")[:32767]


class DeltaMutation(object):

    def __init__(self, inp, atts):
        self.inp_type = str(inp.GetType())
        #self.mut_type = str(typ)
        self.atts = copy.copy(atts)

    def __str__(self):

        r = ["input=" + self.inp_type, "type=" + self.mut_type]
        r = r + map(lambda a_b: a_b[0] + "=" + str(a_b[1]), self.atts.items())
        return " ".join(r)


class NullDeltaMutation(DeltaMutation):

    def __init__(self):
        # pass
        #DeltaMutation.__init__(inp, atts)
        #super(self.__class__, self).__init__(inp, atts)
        self.mut_type = "null"

    def __str__(self):
        r = ["type=" + self.mut_type]
        return " ".join(r)

    def inv(self):
        pass


class OneByteDeltaMutation(DeltaMutation):

    def __init__(self, inp, atts):
        #DeltaMutation.__init__(inp, atts)
        super(self.__class__, self).__init__(inp, atts)
        self.mut_type = "mod"

    def inv(self):
        t = self.atts["new"]
        self.atts["new"] = self.atts["old"]
        self.atts["old"] = t


class ByteExtensionDeltaMutation(DeltaMutation):

    def __init__(self, inp, atts):
        super(self.__class__, self).__init__(inp, atts)
        self.mut_type = "ext"

    def inv(self):
        self.mut_type = "con"
        t = self.atts["new"]
        self.atts["new"] = self.atts["old"]
        self.atts["old"] = t


class Mutator:

    def __init__(self, input):
        self.i = 0
        self.input = input.copy()
        self.input_len = len(input)

        if isinstance(input, Input.Arg):
            self.array = map(chr, range(1, 256))
        elif isinstance(input, Input.File):
            self.array = map(chr, range(0, 256))

        self.array_len = len(self.array)

    # def GetDelta(self):

    def Mutate(self):
        assert(0)

    def GetData(self):
        return None

    def GetDelta(self):
        assert(0)


class RandomExpanderMutator(Mutator):

    max_expansion = 10000

    def __iter__(self):
        return self

    def next(self):

        assert(self.input_len > 0)

        input = self.input.copy()
        delta = str(self.input.GetType()) + " "

        # expansion mutation
        i = random.randrange(self.input_len)
        j = random.randrange(self.max_expansion)
        m = self.array[random.randrange(self.array_len)]

        # print self.array[rand]
        input.data = input.data[:i] + m * j + input.data[i + 1:]

        rpos = int(i / (float(self.input_len)) * 100.0)
        rsize = j / 100 * 100
        self.delta = ByteExtensionDeltaMutation(input, dict(
            pos=rpos, size=rsize, old=ord(self.input.data[i]), new=ord(m)))

        return input

    def GetInput(self):
        return self.input.copy()

    def GetDelta(self):
        return self.delta


class RandomByteMutator(Mutator):

    def __iter__(self):
        return self

    def next(self):

        assert(self.input_len > 0)

        input = self.input.copy()
        delta = str(self.input.GetType()) + " "

        # single byte mutation
        i = random.randrange(self.input_len)
        #m = self.array[random.randrange(self.array_len)]
        m = ord(input.data[i]) ^ (1 << random.randrange(7))
        input.data = input.data[:i] + chr(m) + input.data[i + 1:]

        rpos = int(i / (float(self.input_len)) * 100.0)
        # OneByteDeltaMutation(input, dict(pos = rpos, old = ord(self.input.data[i]), new=ord(m)))
        self.delta = None
        return input

    def GetInput(self):
        return self.input.copy()

    def GetDelta(self):
        return self.delta


class NullMutator(Mutator):

    def __iter__(self):
        return self

    def next(self):

        input = self.input.copy()
        return input

    def GetInput(self):
        return self.input.copy()

    # def GetData(self):

    def GetDelta(self):
        return NullDeltaMutation()


class RandomInputMutator:

    def __init__(self, inputs, mutator):
        assert(inputs != [])
        self.i = 0
        self.inputs = map(mutator, inputs)
        self.inputs_len = len(self.inputs)
        self.symb_inputs = filter(lambda x: x[1].input.isSymbolic(), enumerate(self.inputs))
        self.symb_inputs_len = len(self.symb_inputs)

    def __iter__(self):
        return self

    def next(self, mutate=True):
        r = []
        delta = None

        self.i = self.symb_inputs[random.randrange(self.symb_inputs_len)][0]

        for j, m in enumerate(self.inputs):
            if self.i == j:
                r.append(m.next())
                #data = input.PrepareData()
                delta = m.GetDelta()

            else:
                r.append(m.GetInput())
                #data = input.PrepareData()

            # if data:
            #  r.append(data)

        return delta, r

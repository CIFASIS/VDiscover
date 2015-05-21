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

import Input

class DeltaMutation(object):
  def __init__(self, inp, atts):
    self.inp_type = str(inp.GetType())
    #self.mut_type = str(typ)
    self.atts = copy.copy(atts) 

  def __str__(self):

    r = ["input="+self.inp_type, "type="+self.mut_type]
    r = r + map(lambda (a,b): a+"="+str(b),self.atts.items())  
    return " ".join(r)


class NullDeltaMutation(DeltaMutation):

  def __init__(self):
    #pass
    #DeltaMutation.__init__(inp, atts)
    #super(self.__class__, self).__init__(inp, atts)
    self.mut_type = "null"

  def __str__(self):
    r = ["type="+self.mut_type]
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

    if   isinstance(input, Input.Arg):
      self.array = map(chr, range(1, 256))
    elif isinstance(input, Input.File):
      self.array = map(chr, range(0, 256))

    self.array_len = len(self.array)

  #def GetDelta(self):

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
    delta = str(self.input.GetType())+" "
    
    # expansion mutation
    i = random.randrange(self.input_len)
    j = random.randrange(self.max_expansion)
    m = self.array[random.randrange(self.array_len)]

    #print self.array[rand]
    input.data = input.data[:i] + m*j + input.data[i+1:]

      
    rpos = int(i/(float(self.input_len))*100.0) 
    rsize = j/100*100
    self.delta = ByteExtensionDeltaMutation(input,  dict(pos = rpos, size = rsize, old = ord(self.input.data[i]), new = ord(m) )) 
 
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
    delta = str(self.input.GetType())+" "
 
    # single byte mutation
    i = random.randrange(self.input_len)
    m = self.array[random.randrange(self.array_len)]
    input.data = input.data[:i] + m + input.data[i+1:]
      
    rpos = int(i/(float(self.input_len))*100.0) 
    self.delta = OneByteDeltaMutation(input, dict(pos = rpos, old = ord(self.input.data[i]), new=ord(m))) 
    return input

  def GetInput(self):
    return self.input.copy()

  def GetDelta(self):
    return self.delta

"""
class SurpriseMutator(Mutator):

  max_expansion = 10000

  def __iter__(self):
    return self

  def next(self):

    assert(self.input_len > 0)

    input = self.input.copy()
    delta = str(self.input.GetType())+" "
 
    m = random.sample(["s","e"],1)[0]
    #delta = delta 

    if "s" in m:
      # single byte mutation
      i = random.randrange(self.input_len)
      m = self.array[random.randrange(self.array_len)]
      input.data = input.data[:i] + m + input.data[i+1:]
      

      #print i, self.input_len, i/float(self.input_len)
      rpos = int(i/(float(self.input_len))*100.0) 
      delta = OneByteDeltaMutation(input, dict(pos = rpos, old = ord(self.input.data[i]), new=ord(m))) 
      #delta = delta + "mod" + " " + "pos="+str(i) + " " + "old=" + str(ord(self.input.data[i]))+ " " + "new=" + str(ord(m))

    if "e" in m:
      # expansion mutation
      i = random.randrange(self.input_len)
      j = random.randrange(self.max_expansion)
      m = self.array[random.randrange(self.array_len)]
      #delta = delta + "exp" + " " + "pos=" + str(i) + " " + "size=" + str(j) + " " + "old=" + str(ord(self.input.data[i]))+ " " + "new="+ str(ord(m))

      #print self.array[rand]
      input.data = input.data[:i] + m*j + input.data[i+1:]

      
      rpos = int(i/(float(self.input_len))*100.0) 
      rsize = j/100*100
      delta = ByteExtensionDeltaMutation(input,  dict(pos = rpos, size = rsize, old = ord(self.input.data[i]), new = ord(m) )) 

    
    self.delta = delta
    return input

  def GetInput(self):
    return self.input.copy()

  def GetDelta(self):
    return self.delta
"""

class NullMutator(Mutator):

  def __iter__(self):
    return self

  def next(self):

    input = self.input.copy()
    return input

  def GetInput(self):
    return self.input.copy()

  #def GetData(self):

  def GetDelta(self):
    return NullDeltaMutation()


"""class BruteForceMutator(Mutator):

  array_i = 0

  def __iter__(self):
    return self

  def next(self):

    i = self.i
    input = self.input.copy()
    #print self.array[rand]
    input.data = input.data[:i] + self.array[self.array_i] + input.data[i+1:]

    if self.array_i == self.array_len-1:
      self.array_i = 0

      if i == self.input_len-1:
        raise StopIteration
      else:
        self.i = self.i + 1

    else:
      self.array_i = self.array_i + 1

    return input

  def GetInput(self):
    return self.input.copy()

  def GetDelta(self):

    delta = dict()

    delta["aoffset"] = self.i
    delta["roffset"] = (float(self.i) / self.input_len) * 100
    delta["mtype"] = "."

    delta["byte"] = ord(self.array[self.array_i-1])

    delta["iname"] = self.input.GetName()
    delta["itype"] = self.input.GetType()

    return delta

class BruteForceExpander(Mutator):

  array_i = 0
  new_size = 300

  def __iter__(self):
    return self

  def next(self):

    i = self.i
    input = self.input.copy()
    #print self.array[rand]
    input.data = input.data[:i] + self.array[self.array_i]*self.new_size + input.data[i+1:]

    if self.array_i == self.array_len-1:
      self.array_i = 0

      if i == self.input_len-1:
        raise StopIteration
      else:
        self.i = self.i + 1

    else:
      self.array_i = self.array_i + 1

    return input

  def GetInput(self):
    return self.input.copy()

  def GetDelta(self):

    delta = dict()

    delta["aoffset"] = self.i
    delta["roffset"] = (float(self.i) / self.input_len) * 100
    delta["mtype"] = "+"+str(self.new_size)

    delta["byte"] = ord(self.array[self.array_i-1])

    delta["iname"] = self.input.GetName()
    delta["itype"] = self.input.GetType()

    return delta


class InputMutator:
  def __init__(self, args, files, mutator):
    assert(args <> [] or files <> [])
    self.i = 0
    self.arg_mutators  = []
    self.file_mutators = []
    #self.inputs = list(inputs)

    for input in args:
      self.arg_mutators.append(mutator(input))
    for input in files:
      self.file_mutators.append(mutator(input))

    self.inputs = self.arg_mutators + self.file_mutators
    self.inputs_len = len(self.inputs)
  #def __mutate__(self, j,

  def __iter__(self):
    return self

  def next(self, mutate = True):
    r = []
    delta = None

    for j, m in enumerate(self.arg_mutators + self.file_mutators):
      if self.i == j and mutate:
         try:
           input = m.next()
           data = input.PrepareData()
           delta = m.GetDelta()
           #delta = input.GetType(), i, v

         except StopIteration:
           self.i = self.i + 1

           if self.i == self.inputs_len:
             raise StopIteration

           return self.next()

      else:
        input = m.GetInput()
        data = input.PrepareData()

      if data:
        r.append(data)

    return delta, r

  #def GetDelta(self):
  #
  #  mutator = self.inputs[self.i]
  #  input = mutator.GetInput()
  #
  #  offset, val = mutator.GetDelta()
  #  return [input.GetName(), offset, val]:

    #f = lambda m: m.GetInput().PrepareData()

    #args = GetArgs()
    #files = GetFiles()
    #mutator = RandomMutator(args[1])

    #for i in range(0):
    #  mutator.Mutate()

    #args[0] = mutator.Mutate()

    #return " ".join(map(f,self.arg_mutators)) + " " + "".join(map(f,self.file_mutators))
    #+ " > /dev/null 2> /dev/null\""
"""

class RandomInputMutator:
  def __init__(self, inputs, mutator):
    assert(inputs <> [])
    self.i = 0
    self.inputs = map(mutator, inputs)
    self.inputs_len = len(self.inputs)
  
  def __iter__(self):
    return self

  def next(self, mutate = True):
    r = []
    delta = None
    symb_inputs = filter(lambda (_,x): x.input.isSymbolic(), enumerate(self.inputs))
    symb_inputs_len = len(symb_inputs)
    
    self.i = symb_inputs[random.randrange(symb_inputs_len)][0]

    for j, m in enumerate(self.inputs):
      if self.i == j:
        r.append(m.next())
        #data = input.PrepareData()
        delta = m.GetDelta()

      else:
        r.append(m.GetInput()) 
        #data = input.PrepareData()

      #if data:
      #  r.append(data)

    return delta, r


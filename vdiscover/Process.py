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

from ptrace import PtraceError
from ptrace.debugger import (PtraceDebugger, Application,
    ProcessExit, NewProcessEvent, ProcessSignal,
    ProcessExecution, ProcessError)

from logging import getLogger, info, warning, error
from ptrace.error import PTRACE_ERRORS, PtraceError, writeError
from ptrace.disasm import HAS_DISASSEMBLER
from ptrace.ctypes_tools import (truncateWord,
    formatWordHex, formatAddress, formatAddressRange, word2bytes)

from ptrace.signames import signalName, SIGNAMES
from signal import SIGTRAP, SIGALRM, SIGABRT, SIGSEGV, SIGILL, SIGCHLD, SIGWINCH, SIGFPE, SIGBUS, SIGTERM, SIGPIPE, signal, alarm
from errno import ESRCH, EPERM
from ptrace.cpu_info import CPU_POWERPC
from ptrace.debugger import ChildError

from time import sleep

from Event import Exit, Abort, Timeout, Crash, Signal, Call, specs, hash_events

from Vulnerabilities import detect_vulnerabilities
from ELF import ELF
from Run import Launch
from MemoryMap import MemoryMaps
from Alarm import alarm_handler, TimeoutEx

class Process(Application):
    def __init__(self, program, envs, timeout, included_mods = [], ignored_mods = [], no_stdout = True, max_events = 320, min_events = -10*320):

        Application.__init__(self)  # no effect

        self.program = str(program)
        self.name = self.program.split("/")[-1]
        #self.outdir = str(outdir)
        self.no_stdout = no_stdout
        self.envs = envs
        self.timeout = timeout

        self.process = None
        self.included_mods = list(included_mods)
        self.ignored_mods = list(ignored_mods)

        self.pid = None
        self.mm = None
        self.timeouts = 0
        self.max_events = max_events
        self.min_events = min_events

        # Parse ELF
        self.elf = ELF(self.program, plt = False)

        if self.elf.GetType() <> "ELF 32-bit":
          print "Only ELF 32-bit are supported to be executed."
          exit(-1)

        self.modules = dict()

        self.last_signal = {}
        self.last_call = None
        self.crashed = False
        self.nevents = dict()
        self.events = []

        self.binfo = dict()

    def setBreakpoints(self, elf):
      #print elf.GetFunctions()
      for func_name in elf.GetFunctions():
        #print elf.GetModname(), hex(elf.FindFuncInPlt(func_name))

        if func_name in specs:
          #print elf.GetModname(), func_name, hex(elf.FindFuncInPlt(func_name))
          addr = elf.FindFuncInPlt(func_name)
          self.binfo[addr] = elf.GetModname(),func_name
          self.breakpoint(addr)

    def findBreakpointInfo(self, addr):
      if addr in self.binfo:
        return self.binfo[addr]
      else:
        return None, None

    def createEvents(self, signal):
        # Hit breakpoint?
        if signal.signum == SIGTRAP:
            ip = self.process.getInstrPointer()
            if not CPU_POWERPC:
                # Go before "INT 3" instruction
                ip -= 1
            breakpoint = self.process.findBreakpoint(ip)
            #print "breakpoint @",hex(ip)

            if breakpoint:
                module, name = self.findBreakpointInfo(breakpoint.address)
                #print module, name, hex(ip)

                if ip == self.elf.GetEntrypoint():
                  breakpoint.desinstall(set_ip=True)

                  #if self.mm is None:
                  self.mm  = MemoryMaps(self.program, self.pid)
                  #self.setBreakpoints(self.elf)

                  #print self.mm

                  for (range, mod, atts) in self.mm.items():
                     if '/' in mod and 'x' in atts and not ("libc-" in mod):

                        # FIXME: self.elf.path should be absolute
                        if mod == self.elf.path:
                           base = 0
                        else:
                           base = range[0]

                        if self.included_mods == [] or any(map(lambda l: l in mod, self.included_mods)):
                          if self.ignored_mods == [] or not (any(map(lambda l: l in mod, self.ignored_mods))):

                            if not (mod in self.modules):
                              self.modules[mod] = ELF(mod, base = base)
                            #print "hooking", mod, hex(base)

                            self.setBreakpoints(self.modules[mod])


                  return []

                elif name is None:
                  assert(0)

                else:
                  call = Call(name, module)
                  #self.mm.update()
                  #print "updated mm"
                  call.detect_parameters(self.process, self.mm)
                  breakpoint.desinstall(set_ip=True)

                  call_ip = ip
                  self.process.singleStep()
                  self.debugger.waitProcessEvent()

                  n = self.nevents.get((ip,name), 0)
                  self.nevents[(ip, name)] = n + 2
 
                  for ((ip_,name_),n) in self.nevents.items():

                    if n > self.min_events + 1:
                      self.nevents[(ip_, name_)] = n - 1
                    elif n == self.min_events + 1:
                       self.nevents[(ip_, name_)] = self.min_events
                       #print "restoring!", (ip, name)
                       self.breakpoint(call_ip)

                  if n < self.max_events:
                    self.breakpoint(call_ip)
                  #else:
                    #print "disabled!", (ip, name)
 
                  #print "call detected!"
                  return [call]

        elif signal.signum == SIGABRT:
          self.crashed = True
          return [Signal("SIGABRT",self.process, self.mm), Abort(self.process, self.mm)]

        elif signal.signum == SIGSEGV:
          self.crashed = True
          self.mm  = MemoryMaps(self.program, self.pid)
          return [Signal("SIGSEGV", self.process, self.mm), Crash(self.process, self.mm)]

        elif signal.signum == SIGILL:
          #self.crashed = True
          self.mm  = MemoryMaps(self.program, self.pid)
          return [Signal("SIGILL", self.process, self.mm)]

        elif signal.signum == SIGFPE:
          self.crashed = True
          self.mm  = MemoryMaps(self.program, self.pid)
          return [Signal("SIGFPE", self.process, self.mm), Crash(self.process, self.mm)]

        elif signal.signum == SIGBUS:
          #self.crashed = True
          self.mm  = MemoryMaps(self.program, self.pid)
          return [Signal("SIGBUS", self.process, self.mm)]

        elif signal.signum == SIGCHLD:
          #self.crashed = True
          self.mm  = MemoryMaps(self.program, self.pid)
          return [Signal("SIGCHLD", self.process, self.pid)]

        elif signal.signum == SIGTERM: # killed by the kernel?
          self.crashed = True
          return []

        # Harmless signals
        elif signal.signum == SIGPIPE:
          return [] # User generated, ignore.

        # Harmless signals
        elif signal.signum == SIGWINCH:
          return [] # User generated, ignore.

        else:
          print "I don't know what to do with this signal:", str(signal)
          assert(False)

        return []

    def DetectVulnerabilities(self, preevents, events):
      return detect_vulnerabilities(preevents, events, self.process, self.mm)


    def createProcess(self, cmd, envs, no_stdout):

        self.pid = Launch(cmd, no_stdout, envs)
        #self.ofiles = list(files)
        is_attached = True

        try:
            #print "initial processes:"
            #for p in self.debugger:
            #  print "p:", p
            #print "end processes"
            return self.debugger.addProcess(self.pid, is_attached=is_attached)
        except (ProcessExit, PtraceError), err:
            if isinstance(err, PtraceError) \
            and err.errno == EPERM:
                error("ERROR: You are not allowed to trace process %s (permission denied or process already traced)" % self.pid)
            else:
                error("ERROR: Process can no be attached! %s" % err)
        return None

    def destroyProcess(self, signum, frame):
        assert(self.process is not None)

    def _continueProcess(self, process, signum=None):
        if not signum and process in self.last_signal:
            signum = self.last_signal[process]

        if signum:
            error("Send %s to %s" % (signalName(signum), process))
            process.cont(signum)
            try:
                del self.last_signal[process]
            except KeyError:
                pass
        else:
            process.cont()

    def cont(self, signum=None):

        for process in self.debugger:
            process.syscall_state.clear()
            if process == self.process:
                self._continueProcess(process, signum)
            else:
                self._continueProcess(process)

        # Wait for a process signal
        signal = self.debugger.waitSignals()
        process = signal.process
        events = self.createEvents(signal)
        
        #vulns = self.DetectVulnerabilities(self.events, events)
        #print "vulns detected"
        self.events = self.events + events #+ vulns
        #self.nevents = self.nevents + len(events)


    def readInstrSize(self, address, default_size=None):
        if not HAS_DISASSEMBLER:
            return default_size
        try:
            # Get address and size of instruction at specified address
            instr = self.process.disassembleOne(address)
            return instr.size
        except PtraceError, err:
            warning("Warning: Unable to read instruction size at %s: %s" % (
                formatAddress(address), err))
            return default_size

    def breakpoint(self, address):

        # Create breakpoint
        size = self.readInstrSize(address)
        try:
            bp = self.process.createBreakpoint(address, size)
        except PtraceError, err:
            return "Unable to set breakpoint at %s: %s" % (
                formatAddress(address), err)
        #error("New breakpoint: %s" % bp)
        return None

    def runProcess(self, cmd):

        #print "Running", cmd

        signal(SIGALRM, alarm_handler)

        #if self.pid is None:
        #  timeout = 20*self.timeout
        #else:
        timeout = 10*self.timeout

        alarm(timeout)

        # Create new process
        try:
            self.process = self.createProcess(cmd, self.envs, self.no_stdout)
            self.process.no_frame_pointer = self.elf.no_frame_pointer
            #self.mm  = MemoryMaps(self.program, self.pid)
            #print self.mm
            self.crashed = False
        except ChildError, err:
            print "a"
            writeError(getLogger(), err, "Unable to create child process")
            return
        except OSError, err:
            print "b"
            writeError(getLogger(), err, "Unable to create child process")
            return

        except IOError, err:
            print "c"
            writeError(getLogger(), err, "Unable to create child process")
            return

        if not self.process:
            return


        # Set the breakpoints
        self.breakpoint(self.elf.GetEntrypoint())
        #print hex(self.elf.GetEntrypoint())

        try:
          while True:

            #self.cont() 
            #if self.nevents > self.max_events:
            #
            #    self.events.append(Timeout(timeout))
            #    alarm(0)
            #    return
            if not self.debugger or self.crashed:
                # There is no more process: quit
                alarm(0)
                return
            else:
              self.cont()

          #alarm(0)
        #except PtraceError:
          #print "deb:",self.debugger, "crash:", self.crashed
          #print "PtraceError"
          #alarm(0)
          #return        

        except ProcessExit, event:
          alarm(0)
          self.events.append(Exit(event.exitcode))
          return

        except OSError:
          alarm(0)
          self.events.append(Timeout(timeout))
          self.timeouts += 1
          return

        except IOError:
          alarm(0)
          self.events.append(Timeout(timeout))
          self.timeouts += 1
          return

        except TimeoutEx:
          self.events.append(Timeout(timeout))
          return



    def getData(self, inputs):
        self.events = []
        self.nevents = dict()
        self.debugger = PtraceDebugger()

        self.runProcess([self.program]+inputs)
        #print self.pid

        #if self.crashed:
        #  print "we should terminate.."
        #sleep(3)

        if self.process is None:
          return None

        self.process.terminate()
        self.process.detach()
        #print self.nevents

        self.process = None
        return self.events

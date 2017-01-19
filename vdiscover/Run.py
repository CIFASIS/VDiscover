# -- coding: utf-8 --
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


#from ptrace.debugger.child import createChild
from os import system, dup2, close, open as fopen, O_RDONLY
from sys import stdin
from os import (
    fork, execv, execve, getpid,
    close, dup2, devnull, O_RDONLY)

from ptrace.binding import ptrace_traceme
from ptrace import PtraceError

from resource import getrlimit, setrlimit, RLIMIT_AS
fds = []
c = 0


class ChildError(RuntimeError):
    pass


def _execChild(arguments, no_stdout, env):
    if no_stdout:
        try:
            null = open(devnull, 'wb')
            dup2(null.fileno(), 1)
            dup2(1, 2)
            null.close()
        except IOError as err:
            close(2)
            close(1)
    try:
        if env is not None:
            execve(arguments[0], arguments, env)
        else:
            execv(arguments[0], arguments)
    except Exception as err:
        raise ChildError(str(err))


def createChild(arguments, no_stdout, env=None):
    """
    Create a child process:
     - arguments: list of string where (eg. ['ls', '-la'])
     - no_stdout: if True, use null device for stdout/stderr
     - env: environment variables dictionary

    Use:
     - env={} to start with an empty environment
     - env=None (default) to copy the environment
    """

    # Fork process
    pid = fork()
    if pid:
        return pid
    else:
        # print "limit",getrlimit(RLIMIT_DATA)
        setrlimit(RLIMIT_AS, (1024 * 1024 * 1024, -1))
        # print "limit",getrlimit(RLIMIT_DATA)

        try:
            ptrace_traceme()
        except PtraceError as err:
            raise ChildError(str(err))

        _execChild(arguments, no_stdout, env)
        exit(255)


def Launch(cmd, no_stdout, env):
    global fds
    global c
    c = c + 1
    #cmd = ["/usr/bin/timeout", "-k", "1", "3"]+cmd
    # print cmd
    if cmd[-1][0:2] == "< ":
        filename = cmd[-1].replace("< ", "")

        # try:
        #  close(3)
        # except OSError:
        #  print "OsError!"
        #  pass

        for fd in fds:
            # print fd,
            try:
                close(fd)
                # print "closed!"
            except OSError:
                # print "failed close!"
                pass

        fds = []

        desc = fopen(filename, O_RDONLY)
        fds.append(desc)
        dup2(desc, stdin.fileno())
        fds.append(desc)
        # close(desc)

        cmd = cmd[:-1]

    # print "c:", c
    # print "self pid", getpid()

    r = createChild(cmd, no_stdout, env)

    # print "new pid", r
    # print "self pid", getpid()
    # print "Done!"

    return r


# class Runner:
#    def __init__(self, cmd, timeout):
#        #threading.Thread.__init__(self)
#
#        self.cmd = cmd
#        self.timeout = timeout
#
#    def Run(self):
#        #print self.cmd
#        self.p = subprocess.call(self.cmd, shell=False)
#        #self.p.wait()
#        #self.join(self.timeout)
#
#        #if self.is_alive():
#            #print "terminate: ", self.p.pid
#            #self.p.kill()
#            #self.join()
#            #return True
#        return True

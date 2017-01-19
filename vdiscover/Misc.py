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

import socket
import re
import os
import stat
import errno
import string
import base64


def readmodfile(modfile):
    hooked_mods = []
    if modfile is not None:
        hooked_mods = open(modfile).read().split("\n")
        hooked_mods = filter(lambda x: x != '', hooked_mods)
    return hooked_mods

"""
from pwntools
"""


def parse_ldd_output(output):
    """Parses the output from a run of 'ldd' on a binary.
    Returns a dictionary of {path: address} for
    each library required by the specified binary.

    Arguments:
      output(str): The output to parse

    Example:
        >>> sorted(parse_ldd_output('''
        ...     linux-vdso.so.1 =>  (0x00007fffbf5fe000)
        ...     libtinfo.so.5 => /lib/x86_64-linux-gnu/libtinfo.so.5 (0x00007fe28117f000)
        ...     libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fe280f7b000)
        ...     libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fe280bb4000)
        ...     /lib64/ld-linux-x86-64.so.2 (0x00007fe2813dd000)
        ... ''').keys())
        ['/lib/x86_64-linux-gnu/libc.so.6', '/lib/x86_64-linux-gnu/libdl.so.2', '/lib/x86_64-linux-gnu/libtinfo.so.5', '/lib64/ld-linux-x86-64.so.2']
    """
    expr_linux = re.compile(r'\s(?P<lib>\S?/\S+)\s+\((?P<addr>0x.+)\)')
    expr_openbsd = re.compile(
        r'^\s+(?P<addr>[0-9a-f]+)\s+[0-9a-f]+\s+\S+\s+[01]\s+[0-9]+\s+[0-9]+\s+(?P<lib>\S+)$')
    libs = {}

    for s in output.split('\n'):
        match = expr_linux.search(s) or expr_openbsd.search(s)
        if not match:
            continue
        lib, addr = match.group('lib'), match.group('addr')
        libs[lib] = int(addr, 16)

    return libs


def sh_string(s):
    """Outputs a string in a format that will be understood by /bin/sh.

    If the string does not contain any bad characters, it will simply be
    returned, possibly with quotes. If it contains bad characters, it will
    be escaped in a way which is compatible with most known systems.

    Examples:

        >>> print sh_string('foobar')
        foobar
        >>> print sh_string('foo bar')
        'foo bar'
        >>> print sh_string("foo'bar")
        "foo'bar"
        >>> print sh_string("foo\\\\bar")
        'foo\\bar'
        >>> print sh_string("foo\\\\'bar")
        "foo\\\\'bar"
        >>> print sh_string("foo\\x01'bar")
        "$( (echo Zm9vASdiYXI=|(base64 -d||openssl enc -d -base64)||echo -en 'foo\\x01\\x27bar') 2>/dev/null)"
        >>> print subprocess.check_output("echo -n " + sh_string("foo\\\\'bar"), shell = True)
        foo\\'bar
    """

    very_good = set(string.ascii_letters + string.digits)
    good = (very_good | set(string.punctuation + ' ')) - set("'")
    alt_good = (very_good | set(string.punctuation + ' ')) - set('!')

    if '\x00' in s:
        log.error("sh_string(): Cannot create a null-byte")

    if all(c in very_good for c in s):
        return s
    elif all(c in good for c in s):
        return "'%s'" % s
    elif all(c in alt_good for c in s):
        fixed = ''
        for c in s:
            if c in '"\\$`':
                fixed += '\\' + c
            else:
                fixed += c
        return '"%s"' % fixed
    else:
        fixed = ''
        for c in s:
            if c == '\\':
                fixed += '\\\\'
            elif c in good:
                fixed += c
            else:
                fixed += '\\x%02x' % ord(c)
        return '"$( (echo %s|(base64 -d||openssl enc -d -base64)||echo -en \'%s\') 2>/dev/null)"' % (
            base64.b64encode(s), fixed)

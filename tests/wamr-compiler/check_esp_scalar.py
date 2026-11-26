#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test the Espressif LLVM 18 backend using wamrc, a WASI clang and ESP objdump."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def compile_source(clang, source, wasm):
    subprocess.run([clang, '--target=wasm32', '-O2', '-nostdlib', '-Wl,--no-entry',
                    '-Wl,--export-all', '-x', 'c', '-', '-o', wasm],
                   input=source, text=True, check=True)


def instructions(args, work, wasm, features):
    obj = work / 'test.o'
    subprocess.run([args.wamrc, '--target=riscv32', '--target-abi=ilp32f',
                    '--cpu=generic-rv32', '--cpu-features=+m,+a,+f,+c' + features,
                    '--opt-level=3', '--size-level=3', '--format=object', '-o', obj, wasm],
                   stdout=subprocess.DEVNULL, check=True)
    listing = subprocess.check_output([args.objdump, '-d', '-M', 'max,no-aliases', obj],
                                      text=True)
    return re.findall(r'^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)', listing, re.M)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wamrc', 'clang', 'objdump'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        wasm = work / 'test.wasm'
        compile_source(args.clang, """
            unsigned sub2(unsigned a, unsigned b) { return a - (b << 1); }
            unsigned sub4(unsigned a, unsigned b) { return a - (b << 2); }
            unsigned swap(unsigned a) { return __builtin_bswap32(a); }
        """, wasm)
        ops = ('esp.subx2', 'esp.subx4', 'rev8')
        for flags, expected in [('', (0, 0, 0)), (',+zbb', (0, 0, 1)),
                                (',+xesppie', (1, 1, 0)), (',+xesppie,+zbb', (1, 1, 1)),
                                (',+xesppie,-xesppie,+zbb', (0, 0, 1))]:
            listing = instructions(args, work, wasm, flags)
            actual = tuple(listing.count(op) for op in ops)
            assert actual == expected, (flags, actual, expected)
        compile_source(args.clang, """
            unsigned baseline(unsigned x) { return __builtin_popcount(x); }
            __attribute__((export_name("__wamr_rv32_zbb_count"), noinline))
            unsigned variant(unsigned x) { return __builtin_popcount(x); }
        """, wasm)
        assert instructions(args, work, wasm, '').count('cpop') == 1
        # A second input in the rotate/XOR expansion is not a byte swap.
        compile_source(args.clang, """
            unsigned mixed(unsigned x, unsigned y) {
                return __builtin_rotateleft32(x, 24) ^
                       (((__builtin_rotateleft32(x, 16) ^ y) >> 8) & 0x00ff00ff);
            }
            unsigned wrong_scale(unsigned x, unsigned y) { return x - (y << 3); }
        """, wasm)
        listing = instructions(args, work, wasm, ',+xesppie,+zbb')
        assert all(op not in listing for op in ops), listing
    print('RV32 PIE/Zbb instruction selection passed')


if __name__ == '__main__':
    main()

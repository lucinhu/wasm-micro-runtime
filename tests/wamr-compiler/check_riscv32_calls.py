#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compare two RV32 AOTs built from the same Wasm and flags, before/after binding calls."""

import argparse
import re
import struct
from pathlib import Path


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def aligned(size, alignment):
    return (size + alignment - 1) & -alignment


def sections(data):
    assert data[:4] == b"\0aot"
    result = []
    offset = 8
    while offset < len(data):
        kind, size = struct.unpack_from("<II", data, offset)
        start = offset + 8
        assert start + size <= len(data)
        result.append((kind, data[start:start + size]))
        offset = aligned(start + size, 4)
    return result


def relocations(data):
    count = u32(data, 0)
    pool = 8 + count * 4
    names = []
    for index in range(count):
        name = pool + u32(data, 4 + 4 * index)
        size = struct.unpack_from("<H", data, name)[0]
        names.append(data[name + 2:name + 2 + size].removesuffix(b"\0"))
    offset = aligned(pool + u32(data, pool - 4), 4)
    groups = u32(data, offset)
    offset += 4
    records = []
    for _ in range(groups):
        group, length = struct.unpack_from("<II", data, offset)
        offset += 8
        for _ in range(length):
            site, addend, kind, symbol = struct.unpack_from("<IiII", data, offset)
            records.append((names[group], site, addend, kind, names[symbol]))
            offset += 16
    assert offset == len(data)
    return records


def check(before, after):
    assert before[:8] == after[:8]
    old = sections(before)
    new = sections(after)
    # Includes function offsets, imports, exports, custom sections, and data.
    assert [(k, v) for k, v in old if k not in (2, 5)] == [
        (k, v) for k, v in new if k not in (2, 5)]
    text = bytearray(dict(old)[2])
    candidate = dict(new)[2]
    code_start = aligned(4 + u32(text, 0), 4)
    functions = dict(old)[3]
    kept = iter(relocations(dict(new)[5]))
    next_kept = next(kept, None)
    resolved = 0
    for record in relocations(dict(old)[5]):
        if record == next_kept:
            next_kept = next(kept, None)
            continue
        group, site, addend, kind, symbol = record
        assert group == b".rela.text" and kind in (18, 19) and addend == 0
        match = re.fullmatch(rb"aot_func#([0-9]+)", symbol)
        assert match
        target = u32(functions, int(match[1]) * 4)
        pos = code_start + site
        old_hi, old_lo = struct.unpack_from("<II", text, pos)
        hi, lo = struct.unpack_from("<II", candidate, pos)
        assert old_hi & 0x7f == 0x17 and old_lo & 0x707f == 0x67
        assert old_hi == hi & 0xfff and old_lo == lo & 0xfffff
        assert hi >> 7 & 31 == lo >> 15 & 31 != 0
        # Decode the signed JALR immediate independently of compiler rounding.
        low = struct.unpack("<i", struct.pack("<I", lo))[0] >> 20
        actual = (site + (hi & 0xfffff000) + low) & 0xffffffff
        assert actual == target, (site, symbol, actual, target)
        text[pos:pos + 8] = candidate[pos:pos + 8]
        resolved += 1
    assert resolved and next_kept is None
    assert text == candidate, "unrelated text bytes changed"
    return resolved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    print(f"{check(args.before.read_bytes(), args.after.read_bytes())} RV32 calls verified")

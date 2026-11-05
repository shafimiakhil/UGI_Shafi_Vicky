"""Writes a minimal x64 DLL whose DllMain spins in user mode, a fixed number of iterations, on DLL_PROCESS_DETACH only: a stand-in for the state of process 44928 (a DLL detach handler spinning under LdrShutdownProcess, its exit code already set). The spin is bounded on purpose, so that no test can leave behind a process that never ends. Written by the review of 13.8 (workflow wf_4d1a36b3-79f, make_spin_dll.py) and copied here for scripts/guard_run_selftest.py.
python scripts/guard_spin_dll.py <out.dll> <iterations>
"""
import struct
import sys

out, count = sys.argv[1], int(float(sys.argv[2]))
code = bytes.fromhex("85D2" "7510") + b"\x48\xB9" + struct.pack("<Q", count) + bytes.fromhex("4883E901" "75FA") + bytes.fromhex("B801000000" "C3")
# test edx,edx / jnz done / mov rcx,count / loop: sub rcx,1 / jnz loop / done: mov eax,1 / ret

dos = bytearray(64)
dos[0:2] = b"MZ"
struct.pack_into("<I", dos, 60, 64)
coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x2022)
opt = bytearray()
opt += struct.pack("<HBBIII", 0x20B, 14, 0, 0x200, 0, 0)       # magic, linker, SizeOfCode, InitData, UninitData
opt += struct.pack("<II", 0x1000, 0x1000)                      # entry point, BaseOfCode
opt += struct.pack("<QII", 0x0000000700000000, 0x1000, 0x200)  # ImageBase, SectionAlignment, FileAlignment
opt += struct.pack("<HHHHHH", 6, 0, 0, 0, 6, 0)                # OS, image, subsystem versions
opt += struct.pack("<IIII", 0, 0x2000, 0x200, 0)               # Win32Version, SizeOfImage, SizeOfHeaders, CheckSum
opt += struct.pack("<HH", 2, 0x0100)                           # subsystem GUI, DllCharacteristics NX_COMPAT
opt += struct.pack("<QQQQ", 0x100000, 0x1000, 0x100000, 0x1000)
opt += struct.pack("<II", 0, 16)                               # LoaderFlags, NumberOfRvaAndSizes
opt += bytes(16 * 8)
assert len(opt) == 240, len(opt)
sec = b".text\0\0\0" + struct.pack("<IIIIIIHHI", 0x1000, 0x1000, 0x200, 0x200, 0, 0, 0, 0, 0x60000020)
hdr = bytes(dos) + b"PE\0\0" + coff + bytes(opt) + sec
hdr += bytes(0x200 - len(hdr))
body = code + bytes(0x200 - len(code))
with open(out, "wb") as f:
    f.write(hdr + body)
print("wrote", out, len(hdr) + len(body), "bytes; iterations", count)

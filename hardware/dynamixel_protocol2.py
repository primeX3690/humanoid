"""
Dynamixel Protocol 2.0 packet layer (Robotis) - pure Python: build instruction packets, parse status packets, CRC-16, byte stuffing.
This is the transport-independent part; register ADDRESSES are NOT hard-coded here (they differ per model - take them from the
e-manual of your actuators and put them in the control table you give hardware.hal.DynamixelInterface).

Packet:  FF FF FD 00 | ID | LEN_L LEN_H | INST | PARAM... | CRC_L CRC_H     (LEN = params + 3)
"""
from __future__ import annotations
import struct

HEADER = bytes([0xFF, 0xFF, 0xFD, 0x00])
PING, READ, WRITE, REG_WRITE, ACTION, FACTORY_RESET, REBOOT, CLEAR = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x08, 0x10
STATUS, SYNC_READ, SYNC_WRITE, BULK_READ, BULK_WRITE = 0x55, 0x82, 0x83, 0x92, 0x93
BROADCAST = 0xFE

_CRC_TABLE = None


def _table():
    global _CRC_TABLE
    if _CRC_TABLE is None:
        t = []
        for i in range(256):
            c = i << 8
            for _ in range(8):
                c = ((c << 1) ^ 0x8005) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
            t.append(c)
        _CRC_TABLE = t
    return _CRC_TABLE


def crc16(data: bytes, crc=0) -> int:
    t = _table()
    for b in data:
        crc = ((crc << 8) ^ t[((crc >> 8) ^ b) & 0xFF]) & 0xFFFF
    return crc


def stuff(params: bytes) -> bytes:
    """Insert 0xFD after every FF FF FD sequence in the payload so it cannot be mistaken for a header."""
    out = bytearray()
    for b in params:
        out.append(b)
        if len(out) >= 3 and out[-3:] == bytes([0xFF, 0xFF, 0xFD]): out.append(0xFD)
    return bytes(out)


def unstuff(params: bytes) -> bytes:
    out = bytearray(); i = 0
    while i < len(params):
        out.append(params[i])
        if len(out) >= 3 and out[-3:] == bytes([0xFF, 0xFF, 0xFD]) and i + 1 < len(params) and params[i + 1] == 0xFD: i += 1
        i += 1
    return bytes(out)


def build(id_: int, inst: int, params: bytes = b"") -> bytes:
    p = stuff(params)
    body = HEADER + bytes([id_]) + struct.pack("<H", len(p) + 3) + bytes([inst]) + p
    return body + struct.pack("<H", crc16(body))


def ping(id_): return build(id_, PING)
def read(id_, addr, length): return build(id_, READ, struct.pack("<HH", addr, length))
def write(id_, addr, data: bytes): return build(id_, WRITE, struct.pack("<H", addr) + data)
def sync_read(ids, addr, length): return build(BROADCAST, SYNC_READ, struct.pack("<HH", addr, length) + bytes(ids))


def sync_write(addr, length, per_id: dict):
    """per_id: {id: bytes(length)}"""
    body = struct.pack("<HH", addr, length)
    for i, d in per_id.items():
        assert len(d) == length; body += bytes([i]) + d
    return build(BROADCAST, SYNC_WRITE, body)


ERRORS = {0x01: "result fail", 0x02: "instruction error", 0x03: "CRC error", 0x04: "data range error", 0x05: "data length error",
          0x06: "data limit error", 0x07: "access error"}


class PacketError(Exception): pass


def parse_status(pkt: bytes):
    """-> (id, error_byte, params). Raises PacketError on bad header/CRC/length. Hardware-alert bit is bit 7 of the error byte."""
    if len(pkt) < 11 or pkt[:4] != HEADER: raise PacketError("bad header/length")
    n = struct.unpack("<H", pkt[5:7])[0]
    if len(pkt) != 7 + n: raise PacketError("length mismatch")
    if struct.unpack("<H", pkt[-2:])[0] != crc16(pkt[:-2]): raise PacketError("CRC mismatch")
    if pkt[7] != STATUS: raise PacketError("not a status packet")
    return pkt[4], pkt[8], unstuff(pkt[9:-2])


def decode_error(err_byte):
    return dict(code=err_byte & 0x7F, text=ERRORS.get(err_byte & 0x7F, "ok" if (err_byte & 0x7F) == 0 else "unknown"), hardware_alert=bool(err_byte & 0x80))


def status_packet(id_, params=b"", err=0):                       # used by tests / simulated servos
    p = stuff(params)
    body = HEADER + bytes([id_]) + struct.pack("<H", len(p) + 4) + bytes([STATUS, err]) + p
    return body + struct.pack("<H", crc16(body))


def to_i32(v): return struct.pack("<i", int(v))
def from_i32(b): return struct.unpack("<i", b)[0]

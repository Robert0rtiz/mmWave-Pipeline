# ****************************************************************************
# * (C) Copyright 2020, Texas Instruments Incorporated. - www.ti.com
# ****************************************************************************
# *
# *  Redistribution and use in source and binary forms, with or without
# *  modification, are permitted provided that the following conditions are
# *  met:
# *
# *    Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *
# *    Redistributions in binary form must reproduce the above copyright
# *    notice, this list of conditions and the following disclaimer in the
# *     documentation and/or other materials provided with the distribution.
# *
# *    Neither the name of Texas Instruments Incorporated nor the names of its
# *    contributors may be used to endorse or promote products derived from
# *    this software without specific prior written permission.
# *
# *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# *  PARTICULAR TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# *  A PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  OWNER OR
# *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# *  EXEMPLARY, ORCONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# *  LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY, OR TORT (INCLUDING
# *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *

"""Parse processed mmWave UART packets and export readable summaries.

The upper half of this module is the repo's modern parser surface. It turns the
binary TLV stream into typed Python objects and exposes helpers used by the
notebook and capture pipeline.

The lower half retains TI-compatible helper functions so legacy packet-at-a-time
debugging still works when needed.
"""

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

TC_PASS = 0
TC_FAIL = 1

MAGIC_WORD = bytes((2, 1, 4, 3, 6, 5, 8, 7))
HEADER_STRUCT = struct.Struct("<4H8I")
TLV_HEADER_STRUCT = struct.Struct("<2I")
POINT_STRUCT = struct.Struct("<4f")
SIDE_INFO_STRUCT = struct.Struct("<2H")
STATS_STRUCT = struct.Struct("<6I")

TLV_TYPE_DETECTED_POINTS = 1
TLV_TYPE_RANGE_PROFILE = 2
TLV_TYPE_NOISE_PROFILE = 3
TLV_TYPE_AZIMUT_STATIC_HEAT_MAP = 4
TLV_TYPE_RANGE_DOPPLER_HEAT_MAP = 5
TLV_TYPE_STATS = 6
TLV_TYPE_DETECTED_POINTS_SIDE_INFO = 7
TLV_TYPE_TEMPERATURE_STATS = 9


@dataclass(slots=True)
class PacketHeader:
    """Header fields decoded from one mmWave UART frame packet."""

    version: int
    total_packet_len: int
    platform: int
    frame_number: int
    time_cpu_cycles: int
    num_detected_obj: int
    num_tlvs: int
    subframe_number: int


@dataclass(slots=True)
class DetectedPoint:
    """One detected object with derived polar-angle convenience properties."""

    x: float
    y: float
    z: float
    velocity: float
    snr: int | None = None
    noise: int | None = None

    @property
    def range_m(self) -> float:
        return math.sqrt((self.x * self.x) + (self.y * self.y) + (self.z * self.z))

    @property
    def azimuth_deg(self) -> float:
        return math.degrees(math.atan2(self.x, self.y))

    @property
    def elevation_deg(self) -> float:
        planar = math.sqrt((self.x * self.x) + (self.y * self.y))
        return math.degrees(math.atan2(self.z, planar))


@dataclass(slots=True)
class FrameStats:
    """Runtime stats reported by the demo's stats TLV."""

    inter_frame_processing_time: int
    transmit_output_time: int
    inter_frame_processing_margin: int
    inter_chirp_processing_margin: int
    active_frame_cpu_load: int
    inter_frame_cpu_load: int


@dataclass(slots=True)
class FramePacket:
    """Fully decoded contents of one frame packet from the processed stream."""

    header: PacketHeader
    points: list[DetectedPoint] = field(default_factory=list)
    range_profile: list[int] = field(default_factory=list)
    noise_profile: list[int] = field(default_factory=list)
    stats: FrameStats | None = None
    temperature_report_valid: int | None = None
    temperature_raw: tuple[int, ...] = ()
    tlvs: list[tuple[int, int]] = field(default_factory=list)


def _ensure_bytes(data: bytes | bytearray | Sequence[int], read_num_bytes: int | None = None) -> bytes:
    """Normalize byte-like inputs so the legacy helpers and parser share code."""

    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        raw = bytes(data)
    if read_num_bytes is None:
        return raw
    return raw[:read_num_bytes]


def _parse_header(data: bytes, start_index: int) -> PacketHeader:
    """Decode the fixed-width mmWave packet header at `start_index`."""

    unpacked = HEADER_STRUCT.unpack_from(data, start_index)
    return PacketHeader(
        version=unpacked[4],
        total_packet_len=unpacked[5],
        platform=unpacked[6],
        frame_number=unpacked[7],
        time_cpu_cycles=unpacked[8],
        num_detected_obj=unpacked[9],
        num_tlvs=unpacked[10],
        subframe_number=unpacked[11],
    )


def _parse_points(payload: bytes) -> list[DetectedPoint]:
    """Decode the detected-points TLV payload into typed point objects."""

    if len(payload) % POINT_STRUCT.size != 0:
        raise ValueError("Detected-points TLV length is not aligned to 16-byte point records")
    points: list[DetectedPoint] = []
    for offset in range(0, len(payload), POINT_STRUCT.size):
        points.append(DetectedPoint(*POINT_STRUCT.unpack_from(payload, offset)))
    return points


def _parse_side_info(payload: bytes) -> list[tuple[int, int]]:
    """Decode SNR/noise side-info pairs for detected points."""

    if len(payload) % SIDE_INFO_STRUCT.size != 0:
        raise ValueError("Side-info TLV length is not aligned to 4-byte records")
    side_info: list[tuple[int, int]] = []
    for offset in range(0, len(payload), SIDE_INFO_STRUCT.size):
        side_info.append(SIDE_INFO_STRUCT.unpack_from(payload, offset))
    return side_info


def _parse_uint16_payload(payload: bytes) -> list[int]:
    """Decode TLVs that are stored as packed uint16 bins."""

    if len(payload) % 2 != 0:
        raise ValueError("Expected an even-length uint16 TLV payload")
    if not payload:
        return []
    return list(struct.unpack(f"<{len(payload) // 2}H", payload))


def _parse_stats(payload: bytes) -> FrameStats:
    """Decode the fixed-size stats TLV into a `FrameStats` record."""

    if len(payload) != STATS_STRUCT.size:
        raise ValueError(f"Unexpected stats payload size: {len(payload)}")
    return FrameStats(*STATS_STRUCT.unpack(payload))


def _parse_temperature(payload: bytes) -> tuple[int | None, tuple[int, ...]]:
    """Decode the temperature TLV's validity flag plus raw signed readings."""

    if not payload:
        return None, ()
    if len(payload) < 4:
        raise ValueError("Temperature TLV too short to contain valid flag")
    valid_flag = struct.unpack_from("<I", payload, 0)[0]
    remaining = payload[4:]
    if len(remaining) % 2 != 0:
        return valid_flag, ()
    raw = struct.unpack(f"<{len(remaining) // 2}h", remaining) if remaining else ()
    return valid_flag, raw


def find_magic_index(data: bytes | bytearray | Sequence[int], start_index: int = 0, read_num_bytes: int | None = None) -> int:
    """Locate the next packet magic word inside a byte buffer."""

    raw = _ensure_bytes(data, read_num_bytes)
    return raw.find(MAGIC_WORD, start_index)


def parse_frame_packet(data: bytes | bytearray | Sequence[int], start_index: int = 0, read_num_bytes: int | None = None) -> tuple[FramePacket, int]:
    """Parse exactly one frame packet and return it plus the packet end offset."""

    raw = _ensure_bytes(data, read_num_bytes)
    header_start = find_magic_index(raw, start_index)
    if header_start < 0:
        raise ValueError("Magic word not found")
    if header_start + HEADER_STRUCT.size > len(raw):
        raise ValueError("Incomplete packet header")

    header = _parse_header(raw, header_start)
    packet_end = header_start + header.total_packet_len
    if packet_end > len(raw):
        raise ValueError("Incomplete packet payload")

    frame = FramePacket(header=header)
    cursor = header_start + HEADER_STRUCT.size
    side_info: list[tuple[int, int]] = []

    for _ in range(header.num_tlvs):
        if cursor + TLV_HEADER_STRUCT.size > packet_end:
            raise ValueError("Incomplete TLV header")
        tlv_type, tlv_len = TLV_HEADER_STRUCT.unpack_from(raw, cursor)
        payload_start = cursor + TLV_HEADER_STRUCT.size
        payload_end = payload_start + tlv_len
        if payload_end > packet_end:
            raise ValueError("Incomplete TLV payload")

        payload = raw[payload_start:payload_end]
        frame.tlvs.append((tlv_type, tlv_len))

        # Only the TLVs used by the notebook pipeline are decoded into concrete
        # fields; any others are still recorded in `frame.tlvs` for inspection.
        if tlv_type == TLV_TYPE_DETECTED_POINTS:
            frame.points = _parse_points(payload)
        elif tlv_type == TLV_TYPE_DETECTED_POINTS_SIDE_INFO:
            side_info = _parse_side_info(payload)
        elif tlv_type == TLV_TYPE_RANGE_PROFILE:
            frame.range_profile = _parse_uint16_payload(payload)
        elif tlv_type == TLV_TYPE_NOISE_PROFILE:
            frame.noise_profile = _parse_uint16_payload(payload)
        elif tlv_type == TLV_TYPE_STATS:
            frame.stats = _parse_stats(payload)
        elif tlv_type == TLV_TYPE_TEMPERATURE_STATS:
            frame.temperature_report_valid, frame.temperature_raw = _parse_temperature(payload)

        cursor = payload_end

    for point, (snr, noise) in zip(frame.points, side_info):
        point.snr = snr
        point.noise = noise

    return frame, packet_end


def parse_capture_file(path: str | Path) -> list[FramePacket]:
    """Parse every complete frame packet available in a recorded `.dat` file."""

    raw = Path(path).read_bytes()
    frames: list[FramePacket] = []
    offset = 0
    while True:
        header_start = find_magic_index(raw, offset)
        if header_start < 0:
            break
        if header_start + HEADER_STRUCT.size > len(raw):
            break
        header = _parse_header(raw, header_start)
        if header_start + header.total_packet_len > len(raw):
            break
        frame, packet_end = parse_frame_packet(raw, header_start)
        frames.append(frame)
        offset = packet_end
    return frames


def summarize_capture(frames: Iterable[FramePacket]) -> dict[str, int | float]:
    """Return the lightweight counts used in notebook and handoff summaries."""

    frame_list = list(frames)
    total_points = sum(len(frame.points) for frame in frame_list)
    non_empty_frames = sum(1 for frame in frame_list if frame.points)
    return {
        "frame_count": len(frame_list),
        "non_empty_frame_count": non_empty_frames,
        "total_detected_points": total_points,
        "max_points_in_frame": max((len(frame.points) for frame in frame_list), default=0),
    }


def detections_to_rows(frames: Iterable[FramePacket]) -> list[dict[str, float | int | None]]:
    """Flatten point detections into CSV-friendly row dictionaries."""

    rows: list[dict[str, float | int | None]] = []
    for frame in frames:
        for point_index, point in enumerate(frame.points):
            rows.append(
                {
                    "frame": frame.header.frame_number,
                    "subframe": frame.header.subframe_number,
                    "point_index": point_index,
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                    "velocity": point.velocity,
                    "range_m": point.range_m,
                    "azimuth_deg": point.azimuth_deg,
                    "elevation_deg": point.elevation_deg,
                    "snr": point.snr,
                    "noise": point.noise,
                }
            )
    return rows


def write_detection_csv(frames: Iterable[FramePacket], destination: str | Path) -> Path:
    """Write flattened detections to disk and return the output path."""

    rows = detections_to_rows(frames)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "subframe",
                "point_index",
                "x",
                "y",
                "z",
                "velocity",
                "range_m",
                "azimuth_deg",
                "elevation_deg",
                "snr",
                "noise",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return destination_path


def parse_config_file(path: str | Path) -> dict[str, object]:
    """Parse a TI config file into metadata lines and structured commands."""

    metadata: dict[str, str] = {}
    commands: list[dict[str, object]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("%"):
            content = line.lstrip("% ")
            if ":" in content:
                key, value = content.split(":", 1)
                metadata[key.strip()] = value.strip()
            continue
        parts = line.split()
        commands.append({"command": parts[0], "args": parts[1:]})
    return {"metadata": metadata, "commands": commands}

def getUint32(data):
    """!
       This function coverts 4 bytes to a 32-bit unsigned integer.

        @param data : 1-demension byte array  
        @return     : 32-bit unsigned integer
    """ 
    return (data[0] +
            data[1]*256 +
            data[2]*65536 +
            data[3]*16777216)

def getUint16(data):
    """!
       This function coverts 2 bytes to a 16-bit unsigned integer.

        @param data : 1-demension byte array
        @return     : 16-bit unsigned integer
    """ 
    return (data[0] +
            data[1]*256)

def getHex(data):
    """!
       This function coverts 4 bytes to a 32-bit unsigned integer in hex.

        @param data : 1-demension byte array
        @return     : 32-bit unsigned integer in hex
    """ 
    return bytes(data[::-1]).hex().encode()

def checkMagicPattern(data):
    """!
       This function check if data arrary contains the magic pattern which is the start of one mmw demo output packet.  

        @param data : 1-demension byte array
        @return     : 1 if magic pattern is found
                      0 if magic pattern is not found 
    """ 
    return int(bytes(data[:8]) == MAGIC_WORD)
          
def parser_helper(data, readNumBytes):
    """!
       This function is called by parser_one_mmw_demo_output_packet() function or application to read the input buffer, find the magic number, header location, the length of frame, the number of detected object and the number of TLV contained in this mmw demo output packet.

        @param data                   : 1-demension byte array holds the the data read from mmw demo output. It ignorant of the fact that data is coming from UART directly or file read.  
        @param readNumBytes           : the number of bytes contained in this input byte array  
            
        @return headerStartIndex      : the mmw demo output packet header start location
        @return totalPacketNumBytes   : the mmw demo output packet lenght           
        @return numDetObj             : the number of detected objects contained in this mmw demo output packet          
        @return numTlv                : the number of TLV contained in this mmw demo output packet           
        @return subFrameNumber        : the sbuframe index (0,1,2 or 3) of the frame contained in this mmw demo output packet
    """ 

    raw = _ensure_bytes(data, readNumBytes)
    headerStartIndex = find_magic_index(raw, 0, readNumBytes)
  
    if headerStartIndex == -1: # does not find the magic number i.e output packet header 
        totalPacketNumBytes = -1
        numDetObj           = -1
        numTlv              = -1
        subFrameNumber      = -1
        platform            = -1
        frameNumber         = -1
        timeCpuCycles       = -1
    else: # find the magic number i.e output packet header 
        header = _parse_header(raw, headerStartIndex)
        totalPacketNumBytes = header.total_packet_len
        platform = f"0x{header.platform:08x}".encode()
        frameNumber = header.frame_number
        timeCpuCycles = header.time_cpu_cycles
        numDetObj = header.num_detected_obj
        numTlv = header.num_tlvs
        subFrameNumber = header.subframe_number
        
    print("headerStartIndex    = %d" % (headerStartIndex))
    print("totalPacketNumBytes = %d" % (totalPacketNumBytes))
    print("platform            = %s" % (platform)) 
    print("frameNumber         = %d" % (frameNumber)) 
    print("timeCpuCycles       = %d" % (timeCpuCycles))   
    print("numDetObj           = %d" % (numDetObj)) 
    print("numTlv              = %d" % (numTlv))
    print("subFrameNumber      = %d" % (subFrameNumber))   
                              
    return (headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber)


def parser_one_mmw_demo_output_packet(data, readNumBytes):
    """!
       This function is called by application. Firstly it calls parser_helper() function to find the start location of the mmw demo output packet, then extract the contents from the output packet.
       Each invocation of this function handles only one frame at a time and user needs to manage looping around to parse data for multiple frames.

        @param data                   : 1-demension byte array holds the the data read from mmw demo output. It ignorant of the fact that data is coming from UART directly or file read.  
        @param readNumBytes           : the number of bytes contained in this input byte array  
            
        @return result                : parser result. 0 pass otherwise fail
        @return headerStartIndex      : the mmw demo output packet header start location
        @return totalPacketNumBytes   : the mmw demo output packet lenght           
        @return numDetObj             : the number of detected objects contained in this mmw demo output packet          
        @return numTlv                : the number of TLV contained in this mmw demo output packet           
        @return subFrameNumber        : the sbuframe index (0,1,2 or 3) of the frame contained in this mmw demo output packet
        @return detectedX_array       : 1-demension array holds each detected target's x of the mmw demo output packet
        @return detectedY_array       : 1-demension array holds each detected target's y of the mmw demo output packet
        @return detectedZ_array       : 1-demension array holds each detected target's z of the mmw demo output packet
        @return detectedV_array       : 1-demension array holds each detected target's v of the mmw demo output packet
        @return detectedRange_array   : 1-demension array holds each detected target's range profile of the mmw demo output packet
        @return detectedAzimuth_array : 1-demension array holds each detected target's azimuth of the mmw demo output packet
        @return detectedElevAngle_array : 1-demension array holds each detected target's elevAngle of the mmw demo output packet
        @return detectedSNR_array     : 1-demension array holds each detected target's snr of the mmw demo output packet
        @return detectedNoise_array   : 1-demension array holds each detected target's noise of the mmw demo output packet
    """

    detectedX_array = []
    detectedY_array = []
    detectedZ_array = []
    detectedV_array = []
    detectedRange_array = []
    detectedAzimuth_array = []
    detectedElevAngle_array = []
    detectedSNR_array = []
    detectedNoise_array = []

    result = TC_PASS

    (headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber) = parser_helper(data, readNumBytes)

    if headerStartIndex == -1:
        result = TC_FAIL
        print("************ Frame Fail, cannot find the magic words *****************")
    else:
        raw = _ensure_bytes(data, readNumBytes)
        nextHeaderStartIndex = headerStartIndex + totalPacketNumBytes
        if headerStartIndex + totalPacketNumBytes > readNumBytes:
            result = TC_FAIL
            print("********** Frame Fail, readNumBytes may not long enough ***********")
        elif nextHeaderStartIndex + 8 < readNumBytes and checkMagicPattern(raw[nextHeaderStartIndex:nextHeaderStartIndex+8:1]) == 0:
            result = TC_FAIL
            print("********** Frame Fail, incomplete packet **********")
        elif subFrameNumber > 3:
            result = TC_FAIL
            print("************ Frame Fail, subFrameNumber = %d *****************" % (subFrameNumber))
        else:
            try:
                frame, _ = parse_frame_packet(raw, headerStartIndex)
            except ValueError as exc:
                result = TC_FAIL
                print(f"************ Frame Fail, {exc} *****************")
            else:
                numDetObj = len(frame.points)
                for obj, point in enumerate(frame.points):
                    detectedX_array.append(point.x)
                    detectedY_array.append(point.y)
                    detectedZ_array.append(point.z)
                    detectedV_array.append(point.velocity)
                    detectedRange_array.append(point.range_m)
                    detectedAzimuth_array.append(point.azimuth_deg)
                    detectedElevAngle_array.append(point.elevation_deg)
                    detectedSNR_array.append(point.snr or 0)
                    detectedNoise_array.append(point.noise or 0)
                    print(
                        "    obj%3d: %12f %12f %12f %12f %12f %12f %12f %12d %12d"
                        % (
                            obj,
                            point.x,
                            point.y,
                            point.z,
                            point.velocity,
                            point.range_m,
                            point.azimuth_deg,
                            point.elevation_deg,
                            point.snr or 0,
                            point.noise or 0,
                        )
                    )

    return (result, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedRange_array, detectedAzimuth_array, detectedElevAngle_array, detectedSNR_array, detectedNoise_array)




    










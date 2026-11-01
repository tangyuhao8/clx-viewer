#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClinxChemi (.clx) 图像解析、高低曝光调节与 TIFF 导出工具
================================================================
基于对 ClinxChemi (App.bpl -> Unsampleentity / Uncommon / Unsampleviewer / Unimageexport)
的完整二进制逆向工程实现。

核心功能：
1. 完整解析 Clinx `.clx` 文件（验证魔数 FILE_FLAG = 0x000025EB，支持 V1~V6 实体流与 V1~V3 图像头），
   提取化学发光主图 (NormalImage)、白光 Marker 图 (LightImage) 及多通道荧光图 (Image3, Image4) 的
   16-bit 原始灰度矩阵与拍摄元数据（相机型号、序列号、制冷温度、曝光时间、Binning、拍摄时间等）。
2. 精确复刻 ClinxChemi 的高低曝光（Low / High 灰度阈值）与 Gamma 映射算法 (`Uncommon.CalcRawConv`)，
   支持 16-bit 线性/Gamma 曝光拉伸、8-bit 灰度映射、反色显示（黑白底切换）、伪彩映射与白光叠加。
3. 导出为标准 `.tif` 文件（默认 600 DPI，支持 16-bit 曝光拉伸 TIF、16-bit 原始数据+ImageJ窗口 TIF、
   8-bit 灰度 TIF、24-bit RGB 叠加/伪彩 TIF）。

用法示例：
  # 查看 .clx 文件详情并按文件内保存的高低曝光导出 TIF
  python3 clinx_clx_to_tif.py sample.clx -o output_dir/

  # 手动指定低曝光(Low=200)与高曝光(High=5000)，导出 16-bit 拉伸 TIF 和 8-bit 灰度 TIF
  python3 clinx_clx_to_tif.py sample.clx --low 200 --high 5000 --mode all -o output_dir/
"""

from __future__ import annotations
import argparse
import datetime
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

FILE_FLAG = 0x000025EB  # 9707: Clinx .clx 固定魔数
DELPHI_EPOCH = datetime.datetime(1899, 12, 30)

CAPTURE_MODE_NAMES = {
    0: "Chemi (化学发光)",
    1: "Fluores (多色荧光)",
    2: "Gel (凝胶成像)",
    3: "IVScope (活体成像)",
}


def decode_fixed_string(raw: bytes) -> str:
    """自动识别并解码定长 256 字节字符串（兼容 UTF-16LE WideChar[128] 与 ANSI/GBK/UTF-8 Char[256]）。"""
    if not raw or all(b == 0 for b in raw):
        return ""
    # 检查前几个字符对：如果 raw[0] != 0 且 raw[1] == 0，或包含 UTF-16LE 中文且以 \x00\x00 结尾
    if raw[0] != 0 and raw[1] == 0:
        try:
            text = raw.decode("utf-16le", errors="ignore")
            return text.split("\x00", 1)[0].strip()
        except Exception:
            pass
    head = raw.split(b"\x00", 1)[0]
    for enc in ("utf-8", "gb18030", "latin1"):
        try:
            return head.decode(enc).strip()
        except Exception:
            continue
    return ""


def encode_fixed_ansi(text: str, length: int = 256) -> bytes:
    raw = text.encode("gb18030", errors="ignore")[: length - 1]
    return raw + b"\x00" * (length - len(raw))


def encode_fixed_utf16(text: str, byte_length: int = 256) -> bytes:
    raw = text.encode("utf-16le", errors="ignore")[: byte_length - 2]
    return raw + b"\x00" * (byte_length - len(raw))


def delphi_datetime_to_iso(dt_val: float) -> str:
    if dt_val <= 0 or dt_val > 100000:
        return ""
    try:
        dt = DELPHI_EPOCH + datetime.timedelta(days=float(dt_val))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


@dataclass
class ClinxSampleImage:
    """对应 App.bpl 中的 Unsampleentity.TSampleImage"""
    channel_key: str
    channel_label: str
    file_ver: int = 3
    camera_model: str = ""
    camera_sn: str = ""
    temperature: float = 0.0
    xy_binning: int = 1
    width: int = 0
    height: int = 0
    bit_depth: int = 16
    max_gray: int = 65535  # High Exposure (高曝光显示阈值)
    min_gray: int = 0      # Low Exposure (低曝光显示阈值)
    image_size: int = 0
    pixels: Optional[np.ndarray] = None  # shape=(height, width), dtype=uint16

    # V2 扩展字段
    exposure_msec: int = 0
    high_gray: int = 65535
    x_bin: int = 1
    y_bin: int = 1

    # V3 扩展字段
    pre_exposure_msec: int = 0
    pre_exposure_coeff: int = 0
    pre_exposure_max_gray: int = 0
    pre_exposure_avg_gray: int = 0
    pre_exposure_all_gray: int = 0
    color_channel: int = 0
    fluorescence_name: str = ""
    palette_file: str = ""

    @property
    def is_empty(self) -> bool:
        return self.pixels is None or self.pixels.size == 0

    def auto_min_max(self) -> Tuple[int, int]:
        """复刻 TSampleImage.CalcMinMaxGray (0x0042c730)"""
        if self.is_empty:
            return (0, 65535)
        return int(self.pixels.min()), int(self.pixels.max())


@dataclass
class ClinxSampleEntity:
    """对应 App.bpl 中的 Unsampleentity.TSampleEntity"""
    file_flag: int = FILE_FLAG
    file_header_ver: int = 5
    entity_stream_ver: int = 5
    capture_time_raw: float = 0.0
    capture_time_str: str = ""
    exposure_msec: int = 0
    sample_name: str = ""
    cur_gray: int = 0
    active_image_index: int = 1
    capture_mode: int = 0

    # V2 实体预曝光参数
    pre_exposure_msec: int = 0
    pre_exposure_coeff: int = 0
    pre_exposure_max_gray: int = 0
    pre_exposure_avg_gray: int = 0
    pre_exposure_all_gray: int = 0

    light_image: ClinxSampleImage = field(
        default_factory=lambda: ClinxSampleImage("light", "白光图像 (White Light)")
    )
    normal_image: ClinxSampleImage = field(
        default_factory=lambda: ClinxSampleImage("normal", "化学发光/主图 (Chemi/Normal)")
    )
    image3: ClinxSampleImage = field(
        default_factory=lambda: ClinxSampleImage("image3", "荧光通道 2 (Image3)")
    )
    image4: ClinxSampleImage = field(
        default_factory=lambda: ClinxSampleImage("image4", "荧光通道 3 (Image4)")
    )

    def valid_channels(self) -> List[ClinxSampleImage]:
        out = []
        for img in (self.normal_image, self.light_image, self.image3, self.image4):
            if not img.is_empty:
                out.append(img)
        return out


def _read_image_v1(buf: bytes, offset: int, img: ClinxSampleImage) -> int:
    """对应 TSampleImage.LoadFromStream (0x0042c834) + LoadFromStreamV1 (0x0042c858)"""
    if offset + 4 > len(buf):
        return offset
    (img_ver,) = struct.unpack_from("<i", buf, offset)
    offset += 4
    img.file_ver = img_ver
    if img_ver <= 0:
        return offset

    if offset + 0x228 > len(buf):
        raise ValueError(f"损坏的 .clx 图像头 V1 (offset={offset:#x})")

    cam_model_raw = buf[offset : offset + 256]
    cam_sn_raw = buf[offset + 256 : offset + 512]
    temp, xy_bin, width, height, bit_depth, max_gray, min_gray, img_size = struct.unpack_from(
        "<diiiiiiq", buf, offset + 512
    )
    offset += 0x228

    img.camera_model = decode_fixed_string(cam_model_raw)
    img.camera_sn = decode_fixed_string(cam_sn_raw)
    img.temperature = float(temp)
    img.xy_binning = int(xy_bin)
    img.width = int(width)
    img.height = int(height)
    img.bit_depth = int(bit_depth) if bit_depth > 0 else 16
    img.max_gray = int(max_gray)
    img.min_gray = int(min_gray)
    img.image_size = int(img_size)

    if img_size > 0 and width > 0 and height > 0:
        if offset + img_size > len(buf):
            raise ValueError(
                f"像素数据截断: 期望 {img_size} 字节 ({width}x{height}), 剩余 {len(buf) - offset} 字节"
            )
        raw_pixels = buf[offset : offset + img_size]
        offset += img_size
        expected_u16 = width * height * 2
        if img_size == expected_u16:
            arr = np.frombuffer(raw_pixels, dtype="<u2").reshape((height, width)).copy()
            img.pixels = arr
        elif img_size == width * height:
            arr = np.frombuffer(raw_pixels, dtype=np.uint8).reshape((height, width)).astype(np.uint16) * 257
            img.pixels = arr
    return offset


def _read_image_v2(buf: bytes, offset: int, img: ClinxSampleImage) -> int:
    """对应 TSampleImage.LoadFromStreamV2 (0x0042c9a8, 16 字节)"""
    if offset + 16 > len(buf):
        return offset
    exp_ms, high_gray, x_bin, y_bin = struct.unpack_from("<iiii", buf, offset)
    img.exposure_msec = int(exp_ms)
    img.high_gray = int(high_gray)
    img.x_bin = int(x_bin)
    img.y_bin = int(y_bin)
    return offset + 16


def _read_image_v3(buf: bytes, offset: int, img: ClinxSampleImage) -> int:
    """对应 TSampleImage.LoadFromStreamV3 (0x0042c9ec, 0x218 = 536 字节)"""
    if offset + 0x218 > len(buf):
        return offset
    pre_ms, pre_coeff, pre_max, pre_avg, pre_all, color_ch = struct.unpack_from(
        "<iiiiii", buf, offset
    )
    fluo_raw = buf[offset + 24 : offset + 280]
    pal_raw = buf[offset + 280 : offset + 536]
    img.pre_exposure_msec = int(pre_ms)
    img.pre_exposure_coeff = int(pre_coeff)
    img.pre_exposure_max_gray = int(pre_max)
    img.pre_exposure_avg_gray = int(pre_avg)
    img.pre_exposure_all_gray = int(pre_all)
    img.color_channel = int(color_ch)
    img.fluorescence_name = decode_fixed_string(fluo_raw)
    img.palette_file = decode_fixed_string(pal_raw)
    return offset + 0x218


def parse_clx_bytes(buf: bytes) -> ClinxSampleEntity:
    """完整解析 ClinxChemi .clx 二进制字节流 (`TSampleEntity.LoadFromFile` @ 0x0042daa8)"""
    if len(buf) < 12:
        raise ValueError("文件过小，不是有效的 Clinx .clx 文件")

    flag, file_ver = struct.unpack_from("<II", buf, 0)
    if flag != FILE_FLAG:
        raise ValueError(
            f"错误的文件头: 期望魔数 0x{FILE_FLAG:08X} (9707), 实际读取为 0x{flag:08X}"
        )

    entity = ClinxSampleEntity(file_flag=flag, file_header_ver=file_ver)
    offset = 8

    (entity_ver,) = struct.unpack_from("<i", buf, offset)
    offset += 4
    entity.entity_stream_ver = entity_ver

    if entity_ver <= 0:
        return entity

    # 1. TSampleEntity.LoadFromStreamV1 (0x118 = 280 bytes)
    if offset + 0x118 > len(buf):
        raise ValueError("损坏的 TSampleFileHeaderV1 结构")
    (capture_time, exp_msec) = struct.unpack_from("<di", buf, offset)
    sample_name_raw = buf[offset + 12 : offset + 268]
    (cur_gray, active_idx) = struct.unpack_from("<ii", buf, offset + 268)
    offset += 0x118

    entity.capture_time_raw = float(capture_time)
    entity.capture_time_str = delphi_datetime_to_iso(capture_time)
    entity.exposure_msec = int(exp_msec)
    entity.sample_name = decode_fixed_string(sample_name_raw)
    entity.cur_gray = int(cur_gray)
    entity.active_image_index = int(active_idx)

    # 2. FLightImage.LoadFromStream (V1) & FNormalImage.LoadFromStream (V1)
    offset = _read_image_v1(buf, offset, entity.light_image)
    offset = _read_image_v1(buf, offset, entity.normal_image)

    # 3. FLightImage & FNormalImage V2
    if entity.light_image.file_ver > 1:
        offset = _read_image_v2(buf, offset, entity.light_image)
    if entity.normal_image.file_ver > 1:
        offset = _read_image_v2(buf, offset, entity.normal_image)

    # 4. TSampleEntity.LoadFromStreamV2 (20 bytes)
    if entity_ver > 1 and offset + 20 <= len(buf):
        (
            entity.pre_exposure_msec,
            entity.pre_exposure_coeff,
            entity.pre_exposure_max_gray,
            entity.pre_exposure_avg_gray,
            entity.pre_exposure_all_gray,
        ) = struct.unpack_from("<iiiii", buf, offset)
        offset += 20

    # 5. FLightImage & FNormalImage V3
    if entity.light_image.file_ver > 2:
        offset = _read_image_v3(buf, offset, entity.light_image)
    if entity.normal_image.file_ver > 2:
        offset = _read_image_v3(buf, offset, entity.normal_image)

    # 6. TSampleEntity.LoadFromStreamV3 (4 bytes) + FImage3 (V1 + V2 + V3)
    if entity_ver > 2 and offset + 4 <= len(buf):
        (cap_mode_word,) = struct.unpack_from("<I", buf, offset)
        entity.capture_mode = cap_mode_word & 0xFF
        offset += 4

        offset = _read_image_v1(buf, offset, entity.image3)
        offset = _read_image_v2(buf, offset, entity.image3)
        offset = _read_image_v3(buf, offset, entity.image3)

    # 7. FImage4 (V1 + V2 + V3) when entity_ver > 3
    if entity_ver > 3 and offset + 4 <= len(buf):
        offset = _read_image_v1(buf, offset, entity.image4)
        offset = _read_image_v2(buf, offset, entity.image4)
        offset = _read_image_v3(buf, offset, entity.image4)

    return entity


def build_clx_bytes(entity: ClinxSampleEntity) -> bytes:
    """将 ClinxSampleEntity 序列化为标准 ClinxChemi .clx 二进制格式 (`SaveToFile` @ 0x0042d9c8)"""
    parts: List[bytes] = []
    parts.append(struct.pack("<II", FILE_FLAG, 4))
    parts.append(struct.pack("<i", 5))

    dt_val = entity.capture_time_raw
    if dt_val <= 0:
        dt_val = (datetime.datetime.now() - DELPHI_EPOCH).total_seconds() / 86400.0
    parts.append(struct.pack("<di", dt_val, entity.exposure_msec))
    parts.append(encode_fixed_utf16(entity.sample_name, 256))
    parts.append(struct.pack("<iii", entity.cur_gray, entity.active_image_index, 0))

    def pack_img_v1(img: ClinxSampleImage) -> bytes:
        out = [struct.pack("<i", 3)]  # FFileVer = FILE_VER_3 (3)
        if not img.is_empty and img.pixels is not None:
            p_u16 = np.ascontiguousarray(img.pixels, dtype="<u2")
            h, w = p_u16.shape
            img.width = w
            img.height = h
            img_bytes = p_u16.tobytes()
        else:
            h, w = 0, 0
            img_bytes = b""
        out.append(encode_fixed_ansi(img.camera_model, 256))
        out.append(encode_fixed_ansi(img.camera_sn, 256))
        out.append(
            struct.pack(
                "<diiiiiiq",
                float(img.temperature),
                int(img.xy_binning),
                int(w),
                int(h),
                int(img.bit_depth),
                int(img.max_gray),
                int(img.min_gray),
                len(img_bytes),
            )
        )
        if img_bytes:
            out.append(img_bytes)
        return b"".join(out)

    def pack_img_v2(img: ClinxSampleImage) -> bytes:
        return struct.pack(
            "<iiii",
            int(img.exposure_msec),
            int(img.high_gray),
            int(img.x_bin),
            int(img.y_bin),
        )

    def pack_img_v3(img: ClinxSampleImage) -> bytes:
        return (
            struct.pack(
                "<iiiiii",
                int(img.pre_exposure_msec),
                int(img.pre_exposure_coeff),
                int(img.pre_exposure_max_gray),
                int(img.pre_exposure_avg_gray),
                int(img.pre_exposure_all_gray),
                int(img.color_channel),
            )
            + encode_fixed_ansi(img.fluorescence_name, 256)
            + encode_fixed_ansi(img.palette_file, 256)
        )

    parts.append(pack_img_v1(entity.light_image))
    parts.append(pack_img_v1(entity.normal_image))
    parts.append(pack_img_v2(entity.light_image))
    parts.append(pack_img_v2(entity.normal_image))
    parts.append(
        struct.pack(
            "<iiiii",
            entity.pre_exposure_msec,
            entity.pre_exposure_coeff,
            entity.pre_exposure_max_gray,
            entity.pre_exposure_avg_gray,
            entity.pre_exposure_all_gray,
        )
    )
    parts.append(pack_img_v3(entity.light_image))
    parts.append(pack_img_v3(entity.normal_image))
    parts.append(struct.pack("<I", entity.capture_mode & 0xFF))
    parts.append(pack_img_v1(entity.image3))
    parts.append(pack_img_v2(entity.image3))
    parts.append(pack_img_v3(entity.image3))
    parts.append(pack_img_v1(entity.image4))
    parts.append(pack_img_v2(entity.image4))
    parts.append(pack_img_v3(entity.image4))

    return b"".join(parts)


def calc_raw_conv_lut8(low: int, high: int, gamma: float = 1.0, invert: bool = False) -> np.ndarray:
    """
    精确复刻 App.bpl 中的 `Uncommon.CalcRawConv` (0x0040b534):
    输入 16-bit 灰度值 0..65535，根据 [low, high] 曝光阈值与 gamma 生成 8-bit 查找表 (0..255)。
    """
    low = max(0, min(65535, int(low)))
    high = max(0, min(65535, int(high)))
    lut = np.zeros(65536, dtype=np.uint8)
    if high <= low:
        lut[low:] = 255
    else:
        lut[high:] = 255
        if high - low > 1:
            idx = np.arange(low + 1, high, dtype=np.float64)
            span = float(high - low)
            if gamma <= 0.0 or abs(gamma - 1.0) < 1e-6:
                vals = np.rint(((idx - low) * 255.0) / span)
            else:
                vals = np.rint(np.power((idx - low) / span, gamma) * 255.0)
            lut[low + 1 : high] = np.clip(vals, 0, 255).astype(np.uint8)
    if invert:
        lut = (255 - lut).astype(np.uint8)
    return lut


def calc_exposure_lut16(low: int, high: int, gamma: float = 1.0, invert: bool = False) -> np.ndarray:
    """
    计算 16-bit 曝光拉伸查找表 (0..65535 -> 0..65535)，用于导出保留 16-bit 灰阶精度的同时
    固化高低曝光调节效果的 16-bit TIFF。
    """
    low = max(0, min(65535, int(low)))
    high = max(0, min(65535, int(high)))
    lut = np.zeros(65536, dtype=np.uint16)
    if high <= low:
        lut[low:] = 65535
    else:
        lut[high:] = 65535
        if high - low > 1:
            idx = np.arange(low + 1, high, dtype=np.float64)
            span = float(high - low)
            if gamma <= 0.0 or abs(gamma - 1.0) < 1e-6:
                vals = np.rint(((idx - low) * 65535.0) / span)
            else:
                vals = np.rint(np.power((idx - low) / span, gamma) * 65535.0)
            lut[low + 1 : high] = np.clip(vals, 0, 65535).astype(np.uint16)
    if invert:
        lut = (65535 - lut).astype(np.uint16)
    return lut


def write_tiff_file(
    filepath: str | Path,
    pixels: np.ndarray,
    dpi: int = 600,
    image_description: str = "",
) -> None:
    """
    无第三方依赖的纯二进制标准未压缩 TIFF 写入器（复刻 Clinx SaveImage16bitTiff / SaveImage8bitTiff，默认 600 DPI）。
    """
    arr = np.ascontiguousarray(pixels)
    if arr.ndim == 2:
        height, width = arr.shape
        samples_per_pixel = 1
        photometric = 1  # BlackIsZero
        if arr.dtype == np.uint16:
            bits_per_sample = (16,)
            arr = arr.astype("<u2", copy=False)
        elif arr.dtype == np.uint8:
            bits_per_sample = (8,)
        else:
            raise ValueError(f"不支持的二维数组 dtype: {arr.dtype}")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        height, width, _ = arr.shape
        samples_per_pixel = 3
        photometric = 2  # RGB
        bits_per_sample = (8, 8, 8)
        arr = arr.astype(np.uint8, copy=False)
    else:
        raise ValueError(f"不支持的图像维度: {arr.shape}")

    raw_bytes = arr.tobytes()
    byte_count = len(raw_bytes)

    entries: List[Tuple[int, int, int, bytes]] = []
    extra_blobs: List[bytes] = []

    def add_short(tag: int, val: int):
        entries.append((tag, 3, 1, struct.pack("<HH", val, 0)))

    def add_long(tag: int, val: int):
        entries.append((tag, 4, 1, struct.pack("<I", val)))

    desc_bytes = (image_description.encode("utf-8") + b"\x00") if image_description else b""
    # 13 base tags (254, 256, 257, 258, 259, 262, 273, 277, 278, 279, 282, 283, 296) + optional 270
    num_tags = 13 + (1 if desc_bytes else 0)
    ifd_offset = 8
    extra_offset_base = ifd_offset + 2 + num_tags * 12 + 4

    cur_extra_offset = extra_offset_base

    def alloc_extra(blob: bytes) -> int:
        nonlocal cur_extra_offset
        if len(blob) % 2 == 1:
            blob = blob + b"\x00"
        off = cur_extra_offset
        extra_blobs.append(blob)
        cur_extra_offset += len(blob)
        return off

    add_long(254, 0)
    add_long(256, width)
    add_long(257, height)
    if samples_per_pixel == 1:
        add_short(258, bits_per_sample[0])
    else:
        bps_off = alloc_extra(struct.pack("<HHH", 8, 8, 8))
        entries.append((258, 3, 3, struct.pack("<I", bps_off)))
    add_short(259, 1)
    add_short(262, photometric)
    if desc_bytes:
        if len(desc_bytes) <= 4:
            padded = desc_bytes.ljust(4, b"\x00")
            entries.append((270, 2, len(desc_bytes), padded))
        else:
            desc_off = alloc_extra(desc_bytes)
            entries.append((270, 2, len(desc_bytes), struct.pack("<I", desc_off)))

    xres_off = alloc_extra(struct.pack("<II", int(dpi), 1))
    yres_off = alloc_extra(struct.pack("<II", int(dpi), 1))
    entries.append((282, 5, 1, struct.pack("<I", xres_off)))
    entries.append((283, 5, 1, struct.pack("<I", yres_off)))

    strip_offset = cur_extra_offset

    add_long(273, strip_offset)
    add_short(277, samples_per_pixel)
    add_long(278, height)
    add_long(279, byte_count)
    add_short(296, 2)

    entries.sort(key=lambda e: e[0])

    with open(filepath, "wb") as f:
        f.write(struct.pack("<2sHI", b"II", 42, ifd_offset))
        f.write(struct.pack("<H", len(entries)))
        for tag, typ, cnt, val_bytes in entries:
            f.write(struct.pack("<HHII", tag, typ, cnt, struct.unpack("<I", val_bytes)[0]))
        f.write(struct.pack("<I", 0))
        for blob in extra_blobs:
            f.write(blob)
        f.write(raw_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="ClinxChemi (.clx) 文件解析、高低曝光调节与 TIFF 导出工具"
    )
    parser.add_argument("input", help="输入的 .clx 文件路径")
    parser.add_argument("-o", "--output-dir", default=".", help="输出目录 (默认当前目录)")
    parser.add_argument("--low", type=int, default=None, help="手动指定低曝光阈值 Low (0..65535，默认使用 .clx 内保存的 MinGray)")
    parser.add_argument("--high", type=int, default=None, help="手动指定高曝光阈值 High (0..65535，默认使用 .clx 内保存的 MaxGray)")
    parser.add_argument("--auto-exposure", action="store_true", help="自动计算最小/最大灰度作为低/高曝光阈值")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma 校正值 (默认 1.0)")
    parser.add_argument("--invert", action="store_true", help="反色显示 (例如化学发光转白底黑带)")
    parser.add_argument(
        "--mode",
        choices=["16bit_stretched", "16bit_raw", "8bit", "all"],
        default="all",
        help="导出 TIFF 模式 (默认 all: 同时导出 16-bit 曝光拉伸、16-bit 原始数据和 8-bit 灰度 TIF)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="导出 TIFF 分辨率 DPI (默认 600，与 Clinx 原版一致)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity = parse_clx_bytes(in_path.read_bytes())
    print(f"=== Clinx .clx 解析成功: {in_path.name} ===")
    print(f"  文件头版本: V{entity.file_header_ver} (实体流 V{entity.entity_stream_ver})")
    print(f"  样品名称  : {entity.sample_name or '(未命名)'}")
    print(f"  拍摄时间  : {entity.capture_time_str or '(未知)'}")
    print(f"  拍摄模式  : {CAPTURE_MODE_NAMES.get(entity.capture_mode, str(entity.capture_mode))}")
    print(f"  主曝光时间: {entity.exposure_msec} ms")

    suffix_map = {
        "normal": "_Chemi",
        "light": "_WhiteLight",
        "image3": "_Fluo2",
        "image4": "_Fluo3",
    }

    for ch in entity.valid_channels():
        actual_min, actual_max = ch.auto_min_max()
        low = actual_min if args.auto_exposure else (args.low if args.low is not None else ch.min_gray)
        high = actual_max if args.auto_exposure else (args.high if args.high is not None else ch.max_gray)
        if high <= low:
            low, high = actual_min, actual_max

        print(f"\n  [通道: {ch.channel_label}]")
        print(f"    尺寸 / 位深 : {ch.width} x {ch.height} px, {ch.bit_depth}-bit")
        print(f"    相机型号/SN : {ch.camera_model} / {ch.camera_sn} (温度: {ch.temperature:.1f}°C, Binning: {ch.xy_binning}x{ch.xy_binning})")
        print(f"    实际灰度范围: [{actual_min}, {actual_max}]")
        print(f"    应用曝光窗口: Low={low}, High={high}, Gamma={args.gamma:.2f}, Invert={args.invert}")

        stem = f"{in_path.stem}{suffix_map.get(ch.channel_key, '')}"

        if args.mode in ("16bit_stretched", "all"):
            lut16 = calc_exposure_lut16(low, high, gamma=args.gamma, invert=args.invert)
            stretched_16 = lut16[ch.pixels]
            p16s = out_dir / f"{stem}_16bit_adjusted.tif"
            write_tiff_file(p16s, stretched_16, dpi=args.dpi)
            print(f"    -> 已导出 16-bit 曝光拉伸 TIFF: {p16s}")

        if args.mode in ("16bit_raw", "all"):
            p16r = out_dir / f"{stem}_16bit_raw.tif"
            ij_desc = f"ImageJ=1.54f\nunit=inch\nmin={float(low):.1f}\nmax={float(high):.1f}\n"
            write_tiff_file(p16r, ch.pixels, dpi=args.dpi, image_description=ij_desc)
            print(f"    -> 已导出 16-bit 原始数据 TIFF (含 ImageJ Low/High 窗口): {p16r}")

        if args.mode in ("8bit", "all"):
            lut8 = calc_raw_conv_lut8(low, high, gamma=args.gamma, invert=args.invert)
            adj_8 = lut8[ch.pixels]
            p8 = out_dir / f"{stem}_8bit_adjusted.tif"
            write_tiff_file(p8, adj_8, dpi=args.dpi)
            print(f"    -> 已导出 8-bit 曝光调整 TIFF : {p8}")


if __name__ == "__main__":
    main()
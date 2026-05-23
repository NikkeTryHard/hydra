from __future__ import annotations

import struct
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from hydra_learner.shard_contracts import ACTION_SPACE, PolicyBatch

STREAM_MAGIC = b"HYRMB1\0\0"
FRAME_HEADER = 1
FRAME_BATCH = 2
FRAME_PROGRESS = 3
FRAME_END = 4
DTYPE_F32 = 1
DTYPE_I64 = 2
DTYPE_BOOL = 3
FIELD_OBS = 1
FIELD_ACTIONS = 2
FIELD_LEGAL = 3
FIELD_VALUE = 4
FIELD_GRP = 5
FIELD_ORACLE = 6
FIELD_ORACLE_MASK = 7
FIELD_TENPAI = 8
FIELD_OPP_NEXT = 9
FIELD_DANGER = 10
FIELD_DANGER_MASK = 11
FIELD_SCORE_PDF = 12
FIELD_SCORE_CDF = 13


def _read_exact(stream: Any, size: int) -> bytearray:
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        read = stream.readinto(view[offset:])
        if read is None:
            continue
        if read == 0:
            raise ValueError("truncated raw MJAI frame payload")
        offset += read
    return chunks


def _decode_header(payload: bytes | bytearray, expected_batch_size: int) -> None:
    if len(payload) != 28:
        raise ValueError(f"raw MJAI header length mismatch: {len(payload)}")
    if payload[:8] != STREAM_MAGIC:
        raise ValueError("raw MJAI stream magic mismatch")
    version, batch_size, feature_flags, field_count = struct.unpack_from("<IQII", payload, 8)
    if version != 1:
        raise ValueError(f"unsupported raw MJAI stream version {version}")
    if batch_size != expected_batch_size:
        raise ValueError(f"raw MJAI batch size mismatch: got {batch_size}, expected {expected_batch_size}")
    if feature_flags != 0 or field_count != 13:
        raise ValueError(f"unsupported raw MJAI stream feature_flags={feature_flags} field_count={field_count}")


def _require_payload_bytes(payload: bytes | bytearray, offset: int, size: int, context: str) -> None:
    if offset + size > len(payload):
        raise ValueError(f"truncated raw MJAI batch payload while reading {context}")


def decode_batch(payload: bytes | bytearray) -> PolicyBatch:
    if len(payload) < 16:
        raise ValueError("raw MJAI batch payload too short")
    rows, feature_flags, field_count = struct.unpack_from("<QII", payload, 0)
    if feature_flags != 0:
        raise ValueError(f"unsupported raw MJAI batch feature flags {feature_flags}")
    offset = 16
    fields: dict[int, npt.NDArray[Any]] = {}
    owner = memoryview(payload)
    for _ in range(field_count):
        _require_payload_bytes(payload, offset, 4, "field header")
        field_id, dtype, ndim = struct.unpack_from("<HBB", payload, offset)
        offset += 4
        shape_bytes = 8 * ndim
        _require_payload_bytes(payload, offset, shape_bytes, "field shape")
        shape = struct.unpack_from("<" + "Q" * ndim, payload, offset)
        offset += shape_bytes
        _require_payload_bytes(payload, offset, 8, "field byte length")
        byte_len = struct.unpack_from("<Q", payload, offset)[0]
        offset += 8
        end = offset + byte_len
        if end > len(payload):
            raise ValueError("raw MJAI field exceeds payload length")
        fields[field_id] = _field_array(owner[offset:end], dtype, shape)
        offset = end
    if offset != len(payload):
        raise ValueError("raw MJAI batch payload has trailing bytes")
    return PolicyBatch(
        obs=cast("npt.NDArray[np.float32]", _required(fields, FIELD_OBS, (rows, 192, 34), np.float32)),
        actions=cast("npt.NDArray[np.int64]", _required(fields, FIELD_ACTIONS, (rows,), np.int64)),
        legal_mask=cast("npt.NDArray[np.bool_]", _required(fields, FIELD_LEGAL, (rows, ACTION_SPACE), np.bool_)),
        value_target=cast("npt.NDArray[np.float32]", _required(fields, FIELD_VALUE, (rows,), np.float32)),
        grp_target=cast("npt.NDArray[np.float32]", _required(fields, FIELD_GRP, (rows, 24), np.float32)),
        oracle_target=cast("npt.NDArray[np.float32]", _required(fields, FIELD_ORACLE, (rows, 4), np.float32)),
        oracle_target_mask=cast("npt.NDArray[np.float32]", _required(fields, FIELD_ORACLE_MASK, (rows,), np.float32)),
        tenpai=cast("npt.NDArray[np.float32]", _required(fields, FIELD_TENPAI, (rows, 3), np.float32)),
        opp_next=cast("npt.NDArray[np.float32]", _required(fields, FIELD_OPP_NEXT, (rows, 102), np.float32)),
        danger=cast("npt.NDArray[np.float32]", _required(fields, FIELD_DANGER, (rows, 102), np.float32)),
        danger_mask=cast("npt.NDArray[np.float32]", _required(fields, FIELD_DANGER_MASK, (rows, 102), np.float32)),
        score_pdf=cast("npt.NDArray[np.float32]", _required(fields, FIELD_SCORE_PDF, (rows, 64), np.float32)),
        score_cdf=cast("npt.NDArray[np.float32]", _required(fields, FIELD_SCORE_CDF, (rows, 64), np.float32)),
        safety_target=None,
        safety_mask=None,
    )


def _field_array(data: memoryview, dtype: int, shape: tuple[int, ...]) -> npt.NDArray[Any]:
    if dtype == DTYPE_F32:
        array_dtype = np.dtype("<f4")
    elif dtype == DTYPE_I64:
        array_dtype = np.dtype("<i8")
    elif dtype == DTYPE_BOOL:
        array_dtype = np.dtype(np.bool_)
    else:
        raise ValueError(f"unsupported raw MJAI field dtype {dtype}")
    array = np.frombuffer(data, dtype=array_dtype)
    expected = int(np.prod(shape, dtype=np.int64))
    if array.size != expected:
        raise ValueError(f"raw MJAI field size mismatch: got {array.size}, expected {expected}")
    return array.reshape(shape)


def _required(
    fields: dict[int, npt.NDArray[Any]], field_id: int, shape: tuple[int, ...], dtype: type[np.generic]
) -> npt.NDArray[Any]:
    field = fields.get(field_id)
    if field is None:
        raise ValueError(f"missing raw MJAI field {field_id}")
    if field.shape != shape or field.dtype != np.dtype(dtype):
        raise ValueError(f"raw MJAI field {field_id} contract mismatch: shape={field.shape} dtype={field.dtype}")
    return field

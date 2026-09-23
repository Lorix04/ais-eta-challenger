from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m19_parquet_lite import decode_hybrid


def _varint(x:int)->bytes:
    out=bytearray()
    while True:
        b=x&0x7f; x>>=7
        if x: out.append(b|0x80)
        else: out.append(b); return bytes(out)


def _bitpacked_run(values:list[int], bw:int)->bytes:
    assert len(values)%8==0
    groups=len(values)//8
    bits=[]
    for v in values:
        bits.extend((v>>i)&1 for i in range(bw))
    arr=np.array(bits,dtype=np.uint8)
    body=np.packbits(arr,bitorder='little').tobytes()
    assert len(body)==groups*bw
    return _varint((groups<<1)|1)+body


def test_hybrid_bitpacked_dictionary_indices_above_8_bits_are_row_aligned():
    # Regression for the M19 input-decoder incident: the canonical MMDEC MMSI,
    # timestamp and position dictionaries routinely need >8 index bits.
    bw=13
    vals=[0,1,255,256,511,1023,4095,8191,17,2048,4096,7777,42,7000,300,6000]
    enc=_bitpacked_run(vals,bw)
    got,pos=decode_hybrid(enc,bw,len(vals))
    assert got.tolist()==vals
    assert pos==len(enc)


def test_hybrid_bitpacked_20bit_values_do_not_truncate_to_uint64_group():
    bw=20
    vals=[1,2,3,4,5,6,7,8,1048575,700001,800002,900003,123456,654321,777777,999999]
    got,_=decode_hybrid(_bitpacked_run(vals,bw),bw,len(vals))
    assert got.tolist()==vals


def test_m19_reveal_locator_underscore_column_is_not_accessed_via_itertuples(monkeypatch, tmp_path):
    # Regression for the real-holdout reveal incident: pandas renames leading
    # underscore columns in namedtuples. The implementation must use columns.
    import pandas as pd
    import ais_eta.m19_mmdec_stream as mod
    loc=pd.DataFrame({'sample_key':['x'],'mmsi':[123456789],'decision_time':['2023-07-01T00:00:00Z']})
    # Stop immediately after target map construction by supplying metadata with
    # no row groups. This would previously raise AttributeError first.
    class MD: row_groups=[]
    monkeypatch.setattr(mod,'parse_footer',lambda _:MD())
    try:
        mod.reveal_labels_from_parquet(tmp_path/'dummy.parquet',loc)
    except ValueError as exc:
        assert 'no exact MMDEC SPEC rows' in str(exc)

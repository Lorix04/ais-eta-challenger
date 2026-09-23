"""Small, audited Parquet reader for the canonical MMDEC v1 files.

This fallback exists only because the execution environment used for the M19
external holdout does not ship pyarrow/fastparquet.  It is intentionally narrow:
flat primitive columns, Snappy/none compression, PLAIN and RLE_DICTIONARY pages.
Those are exactly the encodings used by Dataset_AIS_POS.parquet and
Dataset_AIS_SPEC.parquet.

The bit-packed RLE decoder deliberately supports dictionary index widths > 8.
An earlier experimental decoder collapsed each eight-value group into uint64;
that silently corrupted row alignment when the dictionary bit width exceeded
8 bits.  This implementation decodes the little-endian bit stream directly.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
import struct
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
from thrift.Thrift import TType


class _Obj:
    pass


def _read_list(p, cls=None):
    et, n = p.readListBegin(); out = []
    for _ in range(n):
        if et == TType.STRUCT:
            o = cls(); o.read(p); out.append(o)
        elif et == TType.STRING:
            out.append(p.readBinary().decode("utf-8", "replace"))
        elif et == TType.I32:
            out.append(p.readI32())
        elif et == TType.I64:
            out.append(p.readI64())
        else:
            p.skip(et); out.append(None)
    p.readListEnd(); return out


class _SchemaElement(_Obj):
    def __init__(self):
        self.type = self.type_length = self.repetition_type = self.name = self.num_children = self.converted_type = None
    def read(self, p):
        p.readStructBegin()
        while True:
            _, t, i = p.readFieldBegin()
            if t == TType.STOP: break
            if i in (1, 2, 3, 5, 6) and t == TType.I32:
                setattr(self, {1:"type",2:"type_length",3:"repetition_type",5:"num_children",6:"converted_type"}[i], p.readI32())
            elif i == 4 and t == TType.STRING: self.name = p.readString()
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _Statistics(_Obj):
    def __init__(self): self.null_count = None
    def read(self, p):
        p.readStructBegin()
        while True:
            _, t, i = p.readFieldBegin()
            if t == TType.STOP: break
            if i == 3 and t == TType.I64: self.null_count = p.readI64()
            elif i in (1,2,5,6) and t == TType.STRING: p.readBinary()
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _ColumnMetaData(_Obj):
    def __init__(self):
        self.type=self.encodings=self.path_in_schema=self.codec=self.num_values=None
        self.data_page_offset=self.dictionary_page_offset=self.statistics=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i in (1,4) and t==TType.I32: setattr(self,{1:"type",4:"codec"}[i],p.readI32())
            elif i==2 and t==TType.LIST: self.encodings=_read_list(p)
            elif i==3 and t==TType.LIST: self.path_in_schema=_read_list(p)
            elif i in (5,9,11) and t==TType.I64: setattr(self,{5:"num_values",9:"data_page_offset",11:"dictionary_page_offset"}[i],p.readI64())
            elif i==12 and t==TType.STRUCT: self.statistics=_Statistics(); self.statistics.read(p)
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _ColumnChunk(_Obj):
    def __init__(self): self.meta_data=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i==3 and t==TType.STRUCT: self.meta_data=_ColumnMetaData(); self.meta_data.read(p)
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _RowGroup(_Obj):
    def __init__(self): self.columns=[]; self.num_rows=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i==1 and t==TType.LIST: self.columns=_read_list(p,_ColumnChunk)
            elif i==3 and t==TType.I64: self.num_rows=p.readI64()
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _FileMetaData(_Obj):
    def __init__(self): self.version=self.num_rows=self.created_by=None; self.schema=[]; self.row_groups=[]
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i==1 and t==TType.I32: self.version=p.readI32()
            elif i==2 and t==TType.LIST: self.schema=_read_list(p,_SchemaElement)
            elif i==3 and t==TType.I64: self.num_rows=p.readI64()
            elif i==4 and t==TType.LIST: self.row_groups=_read_list(p,_RowGroup)
            elif i==6 and t==TType.STRING: self.created_by=p.readString()
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _DataPageHeader(_Obj):
    def __init__(self): self.num_values=self.encoding=self.definition_level_encoding=self.repetition_level_encoding=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i in (1,2,3,4) and t==TType.I32: setattr(self,{1:"num_values",2:"encoding",3:"definition_level_encoding",4:"repetition_level_encoding"}[i],p.readI32())
            elif i==5 and t==TType.STRUCT: st=_Statistics(); st.read(p)
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _DictionaryPageHeader(_Obj):
    def __init__(self): self.num_values=self.encoding=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i in (1,2) and t==TType.I32: setattr(self,{1:"num_values",2:"encoding"}[i],p.readI32())
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _DataPageHeaderV2(_Obj):
    def __init__(self):
        self.num_values=self.num_nulls=self.num_rows=self.encoding=None
        self.definition_levels_byte_length=self.repetition_levels_byte_length=self.is_compressed=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i in (1,2,3,4,5,6) and t==TType.I32:
                setattr(self,{1:"num_values",2:"num_nulls",3:"num_rows",4:"encoding",5:"definition_levels_byte_length",6:"repetition_levels_byte_length"}[i],p.readI32())
            elif i==7 and t==TType.BOOL: self.is_compressed=p.readBool()
            elif i==8 and t==TType.STRUCT: st=_Statistics(); st.read(p)
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


class _PageHeader(_Obj):
    def __init__(self):
        self.type=self.uncompressed_page_size=self.compressed_page_size=None
        self.data_page_header=self.dictionary_page_header=self.data_page_header_v2=None
    def read(self,p):
        p.readStructBegin()
        while True:
            _,t,i=p.readFieldBegin()
            if t==TType.STOP: break
            if i in (1,2,3) and t==TType.I32: setattr(self,{1:"type",2:"uncompressed_page_size",3:"compressed_page_size"}[i],p.readI32())
            elif i==5 and t==TType.STRUCT: self.data_page_header=_DataPageHeader(); self.data_page_header.read(p)
            elif i==7 and t==TType.STRUCT: self.dictionary_page_header=_DictionaryPageHeader(); self.dictionary_page_header.read(p)
            elif i==8 and t==TType.STRUCT: self.data_page_header_v2=_DataPageHeaderV2(); self.data_page_header_v2.read(p)
            else: p.skip(t)
            p.readFieldEnd()
        p.readStructEnd()


def parse_footer(path: str | Path):
    with Path(path).open("rb") as f:
        f.seek(-8,2); tail=f.read(8); n=struct.unpack("<I",tail[:4])[0]
        if tail[4:] != b"PAR1": raise ValueError("not a Parquet file")
        f.seek(-(8+n),2); b=f.read(n)
    p=TCompactProtocol.TCompactProtocol(TTransport.TMemoryBuffer(b)); md=_FileMetaData(); md.read(p); return md


def _page_header(f, offset):
    f.seek(offset); p=TCompactProtocol.TCompactProtocol(TTransport.TFileObjectTransport(f)); h=_PageHeader(); h.read(p); return h,f.tell()


_NATIVE_SNAPPY = None
_NATIVE_SNAPPY_PROBED = False


def _snappy_lib():
    """Return a native libsnappy handle when available, without probing at import time.

    The primary cross-platform backend is ``python-snappy``.  Native libsnappy
    remains a fallback for minimal Linux environments, but absence of a system
    library must never make importing this module fail (notably on Windows).
    """
    global _NATIVE_SNAPPY, _NATIVE_SNAPPY_PROBED
    if _NATIVE_SNAPPY_PROBED:
        return _NATIVE_SNAPPY
    _NATIVE_SNAPPY_PROBED = True
    candidates = [ctypes.util.find_library("snappy"), "/usr/lib/x86_64-linux-gnu/libsnappy.so.1", "libsnappy.so.1", "libsnappy.so"]
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            lib.snappy_uncompressed_length.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t)]
            lib.snappy_uncompress.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_char_p,ctypes.POINTER(ctypes.c_size_t)]
            _NATIVE_SNAPPY = lib
            break
        except (OSError, AttributeError):
            continue
    return _NATIVE_SNAPPY


def _snappy(data: bytes) -> bytes:
    """Decompress a raw Snappy block using a portable Python backend first."""
    try:
        import snappy as py_snappy
    except ImportError:
        py_snappy = None
    if py_snappy is not None:
        return py_snappy.decompress(data)

    lib = _snappy_lib()
    if lib is None:
        raise RuntimeError(
            "Snappy decompression requires python-snappy (recommended) or a system libsnappy. "
            "Install project requirements, or `pip install python-snappy==0.7.3`."
        )
    src=ctypes.create_string_buffer(data,len(data)); n=ctypes.c_size_t()
    if lib.snappy_uncompressed_length(src,len(data),ctypes.byref(n)): raise RuntimeError("snappy length failure")
    dst=ctypes.create_string_buffer(n.value); m=ctypes.c_size_t(n.value)
    if lib.snappy_uncompress(src,len(data),dst,ctypes.byref(m)): raise RuntimeError("snappy decode failure")
    return dst.raw[:m.value]


def _varint(b,p):
    x=s=0
    while True:
        c=b[p]; p+=1; x|=(c&127)<<s
        if c<128:return x,p
        s+=7


def decode_hybrid(data: bytes | memoryview, bit_width: int, nvalues: int, pos: int = 0):
    """Decode Parquet RLE/bit-packed hybrid values, including bit widths > 8."""
    b=memoryview(data); bw=int(bit_width); n=int(nvalues)
    if n<=0:return np.empty(0,np.int64),pos
    if bw==0:return np.zeros(n,np.int64),pos
    if bw>63: raise NotImplementedError(f"bit width {bw} > 63")
    out=[]; got=0; wb=(bw+7)//8
    while got<n:
        h,pos=_varint(b,pos)
        if h&1==0:
            run=h>>1; v=int.from_bytes(b[pos:pos+wb],"little"); pos+=wb
            k=min(run,n-got); out.append(np.full(k,v,np.int64)); got+=k
        else:
            groups=h>>1; cnt=groups*8; nb=groups*bw
            raw=np.frombuffer(b[pos:pos+nb],np.uint8); pos+=nb
            bits=np.unpackbits(raw,bitorder="little").reshape(groups,8,bw)
            weights=(np.uint64(1)<<np.arange(bw,dtype=np.uint64))
            vals=(bits.astype(np.uint64)*weights).sum(axis=2,dtype=np.uint64).reshape(-1)
            k=min(cnt,n-got); out.append(vals[:k].astype(np.int64,copy=False)); got+=k
    return (np.concatenate(out) if len(out)>1 else out[0]),pos


def _plain(b,t,n,type_length=None):
    mv=memoryview(b)
    if t==1:return np.frombuffer(mv,dtype="<i4",count=n).copy(),4*n
    if t==2:return np.frombuffer(mv,dtype="<i8",count=n).copy(),8*n
    if t==4:return np.frombuffer(mv,dtype="<f4",count=n).copy(),4*n
    if t==5:return np.frombuffer(mv,dtype="<f8",count=n).copy(),8*n
    if t==0:
        nb=(n+7)//8; return np.unpackbits(np.frombuffer(mv,np.uint8,count=nb),bitorder="little")[:n].astype(bool),nb
    if t==6:
        vals=[]; p=0
        for _ in range(n):
            L=struct.unpack_from("<I",mv,p)[0]; p+=4; vals.append(bytes(mv[p:p+L])); p+=L
        return np.array(vals,dtype=object),p
    if t==7:
        L=int(type_length); return np.array([bytes(mv[i*L:(i+1)*L]) for i in range(n)],dtype=object),n*L
    raise NotImplementedError(f"Parquet physical type {t}")


def _leaf_schema(md, path):
    # MMDEC selected columns are top-level flat leaves with unique names.
    leaf=path[-1]
    return next((s for s in md.schema if s.name==leaf and s.type is not None),None)


def decode_column(path: str | Path, md, row_group_index: int, column_index: int):
    cm=md.row_groups[row_group_index].columns[column_index].meta_data
    se=_leaf_schema(md,cm.path_in_schema); maxdef=1 if se and se.repetition_type==1 else 0; tl=se.type_length if se else None
    off=cm.dictionary_page_offset if cm.dictionary_page_offset is not None else cm.data_page_offset
    parts=[]; total=0; dictionary=None
    with Path(path).open("rb") as f:
        while total < cm.num_values:
            h,body_start=_page_header(f,off); f.seek(body_start); body=f.read(h.compressed_page_size); off=body_start+h.compressed_page_size
            if h.type==2:  # DICTIONARY_PAGE
                raw=_snappy(body) if cm.codec==1 else body
                dictionary,_=_plain(raw,cm.type,h.dictionary_page_header.num_values,tl); continue
            if h.type==0:  # DATA_PAGE v1
                raw=_snappy(body) if cm.codec==1 else body; dh=h.data_page_header; n=dh.num_values; p=0
                if maxdef:
                    L=struct.unpack_from("<I",raw,p)[0]; p+=4; lev,_=decode_hybrid(raw[p:p+L],1,n); p+=L; valid=lev==1
                else: valid=np.ones(n,bool)
                nn=int(valid.sum())
                if dh.encoding in (8,2):
                    bw=raw[p]; p+=1; idx,_=decode_hybrid(raw,bw,nn,p); vals=dictionary[idx]
                elif dh.encoding==0: vals,_=_plain(raw[p:],cm.type,nn,tl)
                else: raise NotImplementedError(("encoding",dh.encoding,cm.path_in_schema))
            elif h.type==3:  # DATA_PAGE_V2
                dh=h.data_page_header_v2; n=dh.num_values; rl=dh.repetition_levels_byte_length; dl=dh.definition_levels_byte_length
                valid=(decode_hybrid(body[rl:rl+dl],1,n)[0]==1) if maxdef else np.ones(n,bool); nn=int(valid.sum()); vb=body[rl+dl:]
                raw=_snappy(vb) if cm.codec==1 and (dh.is_compressed is None or dh.is_compressed) else vb; p=0
                if dh.encoding in (8,2):
                    bw=raw[p]; p+=1; idx,_=decode_hybrid(raw,bw,nn,p); vals=dictionary[idx]
                elif dh.encoding==0: vals,_=_plain(raw[p:],cm.type,nn,tl)
                else: raise NotImplementedError(("v2 encoding",dh.encoding,cm.path_in_schema))
            else:
                continue
            if maxdef and nn<n:
                if cm.type in (4,5): arr=np.full(n,np.nan,float)
                elif cm.type in (1,2): arr=np.full(n,np.nan,float)
                else: arr=np.empty(n,dtype=object); arr[:]=None
                arr[valid]=vals
            else: arr=vals
            parts.append(arr); total+=n
    return np.concatenate(parts)


def _column_map(md, rgi: int):
    return {c.meta_data.path_in_schema[-1]:i for i,c in enumerate(md.row_groups[rgi].columns) if len(c.meta_data.path_in_schema)==1}


def iter_flat_row_groups(path: str | Path, columns: Sequence[str]) -> Iterator[pd.DataFrame]:
    """Yield selected *flat* primitive columns one row group at a time."""
    md=parse_footer(path)
    for rgi,rg in enumerate(md.row_groups):
        cmap=_column_map(md,rgi); missing=[c for c in columns if c not in cmap]
        if missing: raise ValueError(f"missing flat columns in Parquet row group {rgi}: {missing}")
        d={c:decode_column(path,md,rgi,cmap[c]) for c in columns}
        lens={len(v) for v in d.values()}
        if lens != {int(rg.num_rows)}:
            raise AssertionError(f"row alignment failure in row group {rgi}: lengths={lens}, expected={rg.num_rows}")
        for c,a in d.items():
            if getattr(a,"dtype",None)==object:
                d[c]=pd.Series(a).map(lambda x: x.decode("utf-8","replace") if isinstance(x,(bytes,bytearray)) else x).to_numpy(object)
        yield pd.DataFrame(d)


def parquet_info(path: str | Path) -> dict:
    md=parse_footer(path)
    return {"rows":int(md.num_rows),"row_groups":len(md.row_groups),"created_by":md.created_by}

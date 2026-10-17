"""Read direct/delay PE imports without executing binaries; audit package closure."""
import argparse
import json
from pathlib import Path
import struct
import prepare_augur_release as R

SYSTEM_DLLS = set(('advapi32 bcrypt cabinet cfgmgr32 comctl32 comdlg32 crypt32 cryptbase cryptsp '
    'bcryptprimitives combase pdh dbghelp dhcpcsvc dnsapi dwmapi gdi32 imagehlp imm32 iphlpapi kernel32 kernelbase mswsock '
    'msvcrt ncrypt netapi32 normaliz ntdll ole32 oleaut32 powrprof propsys psapi rpcrt4 '
    'secur32 setupapi shcore shell32 shlwapi user32 userenv ucrtbase usp10 version winhttp '
    'wininet winmm winscard winspool wintrust wldap32 ws2_32 wtsapi32').split())


def pe_imports(path):
    data=Path(path).read_bytes()
    def u16(p): return struct.unpack_from('<H',data,p)[0]
    def u32(p): return struct.unpack_from('<I',data,p)[0]
    if data[:2]!=b'MZ' or len(data)<64: raise ValueError('not a PE image')
    pe=u32(0x3c)
    if data[pe:pe+4]!=b'PE\0\0': raise ValueError('bad PE signature')
    count=u16(pe+6); opt=pe+24; magic=u16(opt)
    if magic not in (0x10b,0x20b): raise ValueError('unsupported PE optional header')
    base=struct.unpack_from('<Q',data,opt+24)[0] if magic==0x20b else u32(opt+28)
    directory=opt+(112 if magic==0x20b else 96)
    n_dirs=u32(directory-4)
    start=opt+u16(pe+20)
    sections=[]
    for i in range(count):
        p=start+40*i; virtual_size,virtual,raw_size,raw=struct.unpack_from('<IIII',data,p+8)
        sections.append((virtual,max(virtual_size,raw_size),raw,raw_size))
    def offset(rva):
        if 0<=rva<u32(opt+60): return rva
        for virtual,size,raw,raw_size in sections:
            if virtual<=rva<virtual+size and rva-virtual<raw_size: return raw+rva-virtual
        raise ValueError('PE RVA outside raw data')
    def name_at(rva):
        p=offset(rva);end=data.find(b'\0',p,min(p+512,len(data)))
        if end<0: raise ValueError('unbounded PE import name')
        name=data[p:end].decode('ascii').lower()
        if not name or any(c in name for c in '/\\:'): raise ValueError('invalid PE import basename')
        return name
    rows=[]
    for idx,kind,width,name_index in ((1,'direct',20,3),(13,'delay',32,1)):
        if idx>=n_dirs: continue
        rva,size=struct.unpack_from('<II',data,directory+idx*8)
        if not rva: continue
        p=offset(rva)
        for i in range(4096):
            fields=struct.unpack_from('<'+'I'*(width//4),data,p+i*width)
            if not any(fields): break
            value=fields[name_index]
            if kind=='delay' and not fields[0]&1: value-=base
            rows.append(dict(name=name_at(value),kind=kind))
        else: raise ValueError('unbounded PE import descriptors')
    return rows


def audit(root):
    root=Path(root).resolve()
    binaries=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in {'.dll','.pyd','.exe'}]
    provided={}
    for p in binaries: provided.setdefault(p.name.lower(),[]).append(p.relative_to(root).as_posix())
    if not binaries: raise ValueError('no PE binaries to audit')
    rows=[];missing=set()
    for p in binaries:
        imports=pe_imports(p)
        for row in imports:
            name=row['name']
            if name in provided: row.update(provider='package',candidates=provided[name])
            elif name=='nvcuda.dll': row['provider']='nvidia_driver_required'
            elif name.startswith(('api-ms-win-','ext-ms-win-')) or name.removesuffix('.dll') in SYSTEM_DLLS:
                row['provider']='windows_system_required'
            else: row['provider']='unresolved';missing.add(name)
        rows.append(dict(path=p.relative_to(root).as_posix(),sha256=R.sha(p),imports=imports))
    return dict(schema='quasar-windows-pe-import-audit-v1',binaries=rows,unresolved=sorted(missing),
        direct_and_delay_imports=True,scope='static PE imports; runtime LoadLibrary/plugin paths require smoke tests',
        clean_machine_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.root)
    with a.out.open('x',encoding='utf8') as f: json.dump(result,f,indent=2)
    print(json.dumps(dict(binaries=len(result['binaries']),unresolved=result['unresolved'])))
    if result['unresolved']: raise SystemExit(1)

if __name__=='__main__': main()

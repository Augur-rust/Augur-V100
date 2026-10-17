import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import struct
import audit_augur_windows_dependencies as D
import prepare_augur_windows_runtime as W
import augur_portable as P

class RuntimeSafety(unittest.TestCase):
    def test_archive_rejects_escape_alias_and_symlink(self):
        for names in (('../bad',),('C:/bad',),('con.txt',),('x.',),('A','a')):
            with self.subTest(names=names):
                stream=io.BytesIO()
                with zipfile.ZipFile(stream,'w') as z:
                    for n in names: z.writestr(n,b'x')
                stream.seek(0)
                with zipfile.ZipFile(stream) as z,self.assertRaises(ValueError): list(W.members(z))
        self.assertFalse(W.safe_member(r'a\b'))
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            info=zipfile.ZipInfo('link');info.external_attr=0o120777<<16;z.writestr(info,'target')
        stream.seek(0)
        with zipfile.ZipFile(stream) as z,self.assertRaises(ValueError): list(W.members(z))
    def test_exact_pypi_wheel_identity(self):
        ident=dict(name='a_b',version='1.2',sha256='a'*64)
        meta=dict(info=dict(name='a-b',version='1.2'),urls=[dict(filename='a.whl',packagetype='bdist_wheel',digests=dict(sha256='a'*64),size=10,yanked=False,url='https://files.pythonhosted.org/a.whl')])
        self.assertEqual(W.verify_pypi('a.whl',ident,json.dumps(meta),10),meta['urls'][0]['url'])
        for key,value in [('yanked',True),('size',11),('url','https://untrusted.test/a.whl')]:
            bad=json.loads(json.dumps(meta));bad['urls'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): W.verify_pypi('a.whl',ident,json.dumps(bad),10)
    def test_portable_path_does_not_reuse_dev_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'runtime/native').mkdir(parents=True)
            with patch.dict(os.environ,{'CUDA_PATH':'bad','PYTHONPATH':'bad','PATH':'bad'}),patch.object(os,'add_dll_directory',create=True):
                P.activate(root)
                self.assertNotIn('CUDA_PATH',os.environ);self.assertNotIn('PYTHONPATH',os.environ)
                self.assertEqual(os.environ['PATH'].split(os.pathsep)[0],str(root/'runtime/native'))
                self.assertNotIn('bad',os.environ['PATH'])
    def test_missing_runtime_fails_before_model(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(RuntimeError): P.activate(d)

    def test_engine_help_is_forwarded_to_real_launcher(self):
        import sys
        with patch.object(P,'activate',return_value=Path('C:/Augur')),patch.object(P.runpy,'run_path') as run,patch.object(sys,'argv',['portable.py']):
            P.main(['engine','--help'])
            self.assertEqual(sys.argv[-1],'--help');self.assertEqual(sys.argv[1],'--build')
            run.assert_called_once()

class PeInventory(unittest.TestCase):
    def test_direct_and_delay_imports(self):
        data=bytearray(2048);data[:2]=b'MZ';struct.pack_into('<I',data,0x3c,0x80)
        data[0x80:0x84]=b'PE\0\0';struct.pack_into('<H',data,0x86,1);struct.pack_into('<H',data,0x94,240)
        opt=0x98;struct.pack_into('<H',data,opt,0x20b);struct.pack_into('<Q',data,opt+24,0x140000000)
        struct.pack_into('<I',data,opt+60,512);struct.pack_into('<I',data,opt+108,16)
        struct.pack_into('<II',data,opt+112+8,0x1000,40)
        struct.pack_into('<II',data,opt+112+13*8,0x1080,64)
        struct.pack_into('<IIII',data,opt+240+8,1024,0x1000,1024,512)
        struct.pack_into('<IIIII',data,512,0,0,0,0x1040,0)
        data[576:589]=b'KERNEL32.dll\0'
        struct.pack_into('<IIIIIIII',data,640,1,0x10c0,0,0,0,0,0,0)
        data[704:715]=b'nvcuda.dll\0'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fake.exe';p.write_bytes(data)
            self.assertEqual(D.pe_imports(p),[dict(name='kernel32.dll',kind='direct'),dict(name='nvcuda.dll',kind='delay')])
            result=D.audit(tmp);self.assertEqual(result['unresolved'],[])
            self.assertEqual(result['binaries'][0]['imports'][1]['provider'],'nvidia_driver_required')
    def test_invalid_image_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fake.exe';p.write_bytes(b'not PE')
            with self.assertRaises(ValueError):D.pe_imports(p)

    def test_empty_dependency_audit_does_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):D.audit(tmp)

if __name__=='__main__': unittest.main()

"""Portable binary receipts: relocation must not weaken original CPU/source binding."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import package_augur_windows as W
import augur_product as P
import qwen35_chat as C

class PortablePackage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.build=self.root/'build';self.build.mkdir();(self.build/'source').mkdir()
        source=self.build/'source'/'native.rs';source.write_text('pinned source')
        version=self.build/'source/VERSION';version.write_text('v0.0.1-alpha\n')
        binary=self.build/'augur_session_http.exe';binary.write_bytes(b'pinned binary')
        P.save(self.build/'source-identity.json',dict(files={'native.rs':P.sha(source),'VERSION':P.sha(version)}))
        P.save(self.build/'build-receipt.json',dict(feature='augur-model-qwen35-runtime/qwen35-augur-vision',
            source_sha256=P.sha(self.build/'source-identity.json'),
            binaries={'augur_session_http':dict(path=str(binary),sha256=P.sha(binary))}))
        P.save(self.build/'cpu-complete.json',dict(status='PASS',build_sha256=P.sha(self.build/'build-receipt.json')))
        self.out=self.root/'portable'
    def tearDown(self): self.temp.cleanup()
    def modify_mapping(self, action):
        path=self.out/'portable-build.json';value=json.loads(path.read_bytes());action(value)
        path.write_text(json.dumps(value))
    def test_relocation_keeps_original_receipt_bytes_and_uses_only_local_binary(self):
        W.package(self.build,self.out)
        for name in ('build-receipt.json','cpu-complete.json','source-identity.json'):
            self.assertEqual((self.build/name).read_bytes(),(self.out/name).read_bytes())
        moved=self.root/'moved with spaces';shutil.copytree(self.out,moved)
        # Original may still exist, but must never be selected after relocation.
        (self.build/'augur_session_http.exe').write_bytes(b'changed original')
        receipt=P.verified_build(moved)
        self.assertEqual(receipt['binaries']['augur_session_http']['path'],str((moved/'bin/augur_session_http.exe').resolve()))
        (moved/'bin/augur_session_http.exe').write_bytes(b'tampered')
        with self.assertRaises(C.ProtocolError): P.verified_build(moved)
    def test_receipt_and_source_tampering_fail(self):
        W.package(self.build,self.out)
        (self.out/'source/native.rs').write_text('tampered')
        with self.assertRaises(C.ProtocolError): P.verified_build(self.out)
    def test_mapping_cannot_escape_or_add_binary(self):
        W.package(self.build,self.out)
        original=(self.out/'portable-build.json').read_bytes()
        for path in ('../augur_session_http.exe','bin/../outside.exe','C:/outside.exe','bin\\outside.exe','bin//a.exe'):
            with self.subTest(path=path):
                (self.out/'portable-build.json').write_bytes(original)
                self.modify_mapping(lambda v:v['binaries']['augur_session_http'].update(path=path))
                with self.assertRaises(C.ProtocolError): P.verified_build(self.out)
        (self.out/'portable-build.json').write_bytes(original)
        self.modify_mapping(lambda v:v['binaries'].update(extra={'path':'bin/extra.exe','sha256':'a'*64}))
        with self.assertRaises(C.ProtocolError): P.verified_build(self.out)
    def test_mapping_hash_cannot_replace_original_hash(self):
        W.package(self.build,self.out)
        self.modify_mapping(lambda v:v['binaries']['augur_session_http'].update(sha256='a'*64))
        with self.assertRaises(C.ProtocolError): P.verified_build(self.out)
    def test_portable_cpu_binding_remains_enforced(self):
        W.package(self.build,self.out)
        self.modify_mapping(lambda v:v.update(cpu_receipt_sha256='a'*64))
        with self.assertRaises(C.ProtocolError): P.verified_build(self.out)
    def test_no_secrets_logs_or_weights_and_package_requires_new_disjoint_directory(self):
        (self.build/'secret.key').write_text('private key')
        (self.build/'model.gguf').write_bytes(b'weights')
        result=W.package(self.build,self.out)
        self.assertNotIn('secret.key',result['files']);self.assertNotIn('model.gguf',result['files'])
        self.assertFalse(result['product_ready']);self.assertFalse(result['published'])
        with self.assertRaises(FileExistsError): W.package(self.build,self.out)
        with self.assertRaises(ValueError): W.package(self.build,self.build/'child')
    def test_zip_is_exclusive_and_relocatable(self):
        W.package(self.build,self.out)
        archive=self.root/'candidate.zip';receipt=W.archive(self.out,archive)
        self.assertEqual(receipt['sha256'],P.sha(archive))
        with self.assertRaises(FileExistsError): W.archive(self.out,archive)
        with self.assertRaises(ValueError): W.archive(self.out,self.out/'nested.zip')

    def test_source_export_verifies_and_builds_manifest_without_git(self):
        import export_augur_windows_source as S
        import prepare_augur_source_archive as A
        import build_augur_windows as B
        source=self.build/'source/README.md';source.write_text('public source')
        identity=dict(git_head='0'*40,files={'README.md':P.sha(source)})
        (self.build/'source-identity.json').write_text(json.dumps(identity))
        receipt=json.loads((self.build/'build-receipt.json').read_bytes())
        receipt['source_sha256']=P.sha(self.build/'source-identity.json')
        (self.build/'build-receipt.json').write_text(json.dumps(receipt))
        (self.build/'cpu-complete.json').write_text(json.dumps(dict(status='PASS',
            build_sha256=P.sha(self.build/'build-receipt.json'))))
        notices=self.root/'notices';notices.mkdir()
        notice=dict(files={},source_lock_hashes={},source_notice_hashes={})
        (notices/'notice-bundle.json').write_text(json.dumps(notice))
        archive=self.root/'source.zip'
        with patch('prepare_augur_windows_notices.validate',return_value=notice):
            result=S.export(self.build,notices,archive)
        extracted=self.root/'extracted'
        verified=A.verify_archive(archive,P.sha(archive),extracted)
        self.assertEqual(verified,result)
        self.assertEqual(result['source_manifest_sha256'],P.sha(extracted/'augur-release-source.json'))
        self.assertEqual(B.source_inputs(extracted),(identity['files'],identity['git_head']))
        self.assertFalse((extracted/'.git').exists())
        self.assertFalse(result['weights_included'])
        self.assertFalse(result['binaries_included'])

    def test_runtime_bundle_uses_isolated_interpreter_without_system_fallback(self):
        runtime=self.root/'runtime-input';runtime.mkdir();(runtime/'runtime-manifest.json').write_text('{}')
        (runtime/'python').mkdir();py=runtime/'python/python.exe';py.write_bytes(b'python')
        (runtime/'native').mkdir();crt=runtime/'native/vcruntime140.dll';crt.write_bytes(b'crt')
        fake=dict(files={'python/python.exe':dict(sha256=P.sha(py),bytes=6),
                         'native/vcruntime140.dll':dict(sha256=P.sha(crt),bytes=3)})
        with patch('prepare_augur_windows_runtime.validate',return_value=fake):
            result=W.package(self.build,self.out,runtime=runtime)
        self.assertTrue(result['runtime_included'])
        self.assertEqual((self.out/'bin/vcruntime140.dll').read_bytes(),b'crt')
        cmd=(self.out/'Start-Augur-Engine.cmd').read_text()
        self.assertIn('runtime\\python\\python.exe',cmd);self.assertIn('-I -X utf8 -B',cmd)
        self.assertNotIn('AUGUR_PYTHON',cmd);self.assertTrue((self.out/'Verify-Package.cmd').is_file())
    def test_notices_cannot_bind_different_source(self):
        notices=self.root/'notices';notices.mkdir()
        fake=dict(files={},source_lock_hashes={'Cargo.lock':'a'*64},source_notice_hashes={},runtime_manifest_sha256='b'*64)
        with patch('prepare_augur_windows_notices.validate',return_value=fake),self.assertRaisesRegex(ValueError,'notice/source'):
            W.package(self.build,self.out,notices=notices)
        self.assertFalse(self.out.exists())

if __name__=='__main__': unittest.main()

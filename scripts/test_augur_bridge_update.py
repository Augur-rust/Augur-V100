import copy,json,tempfile,unittest
from pathlib import Path
import augur_bridge_entry as E
import package_augur_bridge_update as W


class BridgeUpdate(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.base=self.root/'base';self.overlay=self.root/'overlay';self.out=self.root/'out'
        self.base.mkdir();(self.overlay/'source/scripts').mkdir(parents=True)
        (self.base/'build-receipt.json').write_text('{"original":true}',encoding='utf8')
        (self.base/'native.bin').write_bytes(b'original-native')
        files={p.name:W.sha(p) for p in self.base.iterdir()}
        (self.base/'windows-distribution.json').write_text(json.dumps(dict(files=files)),encoding='utf8')
        (self.overlay/'source/scripts/augur_portable.py').write_text('# overlay entry',encoding='utf8')
        (self.overlay/'private-backup').mkdir();(self.overlay/'private-backup/api-key.txt').write_text('PRIVATE_NOT_PUBLISHABLE')
        (self.overlay/'deployment-manifest.json').write_text(json.dumps(dict(schema='augur-python-bridge-overlay-v1',overlay_files_sha256={'source/scripts/augur_portable.py':W.sha(self.overlay/'source/scripts/augur_portable.py')})),encoding='utf8')

    def package(self):return W.package(self.base,self.overlay,self.out,'v0.0.2-alpha','a'*40)

    def test_preserves_original_bytes_and_excludes_private_backup(self):
        record=self.package();self.assertFalse(record['native_rebuilt']);self.assertFalse(record['research_native_changes_shipped'])
        for name in ['build-receipt.json','native.bin','windows-distribution.json']:
            self.assertEqual((self.base/name).read_bytes(),(self.out/'native-base'/name).read_bytes())
        self.assertFalse(any('private' in p.as_posix() for p in self.out.rglob('*')))
        self.assertEqual(E.verify(self.out),record)

    def test_rejects_file_tampering(self):
        self.package();(self.out/'bridge/source/scripts/augur_portable.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'drift'):E.verify(self.out)

    def test_rejects_unexpected_private_file(self):
        self.package();(self.out/'secret.txt').write_text('do not publish')
        with self.assertRaisesRegex(ValueError,'unexpected'):E.verify(self.out)

    def test_rejects_unlisted_original_identity(self):
        self.package();p=self.out/'bridge-update.json';r=json.loads(p.read_bytes());r['base_distribution_sha256']='0'*64;p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'original distribution'):E.verify(self.out)

    def test_rejects_unsafe_overlay_and_existing_output(self):
        self.package()
        with self.assertRaisesRegex(ValueError,'new disjoint'):self.package()
        p=self.overlay/'deployment-manifest.json';r=json.loads(p.read_bytes());r['overlay_files_sha256']['../private']='0'*64;p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'non-public'):W.package(self.base,self.overlay,self.root/'bad','v0.0.2-alpha','a'*40)

    def test_accepts_verified_windows_manifest_separators(self):
        p=self.overlay/'deployment-manifest.json';r=json.loads(p.read_bytes())
        key='source/scripts/augur_portable.py'
        r['overlay_files_sha256'][key.replace('/', '\\')]=r['overlay_files_sha256'].pop(key)
        p.write_text(json.dumps(r));record=self.package()
        self.assertIn('bridge/source/scripts/augur_portable.py',record['files'])

    def test_safe_names(self):
        for name in ['../x','C:/x','/x','a//b','a\\b','']:
            with self.subTest(name=name):self.assertFalse(E.safe_name(name))
        self.assertTrue(E.safe_name('source/scripts/entry.py'))

if __name__=='__main__':unittest.main()

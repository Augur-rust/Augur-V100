"""Keep the public unsafe boundary contract in the actual release source selection."""
from pathlib import Path
import re
import unittest
import prepare_augur_release as R

class SafetyDocumentation(unittest.TestCase):
    def test_security_contract_is_required_in_public_source(self):
        for name in ('SECURITY.md','docs/UNSAFE_TRUST_BOUNDARIES.md'):
            self.assertTrue(R.selected(name));self.assertIn(name,R.REQUIRED_SOURCE)
            self.assertTrue((R.ROOT/name).is_file())
    def test_contract_keeps_outstanding_gaps_explicit(self):
        text=(R.ROOT/'docs/UNSAFE_TRUST_BOUNDARIES.md').read_text(encoding='utf8')
        for boundary in ('TB-ABI','TB-DEVICE','TB-ASYNC','TB-HOST','TB-GRAPH','TB-STATE','TB-ERROR','TB-FILES',
                         'GAP-VIEW','GAP-GRAPH','GAP-MAPPED','GAP-ERROR'):
            self.assertIn(boundary,text)
        self.assertIn('不是全仓 unsafe 审计通过证明',text)
        self.assertIn('safe 函数',text)
    def test_documentation_links_resolve_to_shipped_inputs(self):
        for name in ('SECURITY.md','docs/UNSAFE_TRUST_BOUNDARIES.md'):
            source=R.ROOT/name
            for target in re.findall(r'\]\(([^)]+)\)',source.read_text(encoding='utf8')):
                self.assertFalse('://' in target)
                path=(source.parent/target.split('#',1)[0]).resolve()
                self.assertTrue(path.is_relative_to(R.ROOT.resolve()))
                self.assertTrue(path.is_file(),target)
                self.assertTrue(R.selected(path.relative_to(R.ROOT).as_posix()),target)
    def test_both_readmes_link_security_and_contract(self):
        for name in ('README.md','README.zh-CN.md'):
            text=(R.ROOT/name).read_text(encoding='utf8')
            self.assertIn('(SECURITY.md)',text)
            self.assertIn('(docs/UNSAFE_TRUST_BOUNDARIES.md)',text)

if __name__=='__main__':unittest.main()

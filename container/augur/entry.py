"""Fixed QUASAR owner entry; weights and trusted admission data are external."""
import argparse
import os
from pathlib import Path
import re
import sys
import uuid

BUILD = Path('/opt/augur/build')
sys.path.insert(0, str(BUILD/'source/scripts'))


def launch_arguments(environment):
    digest = environment.get('AUGUR_MANIFEST_SHA256', '')
    if not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError('trusted AUGUR_MANIFEST_SHA256 is required')
    port = environment.get('AUGUR_PORT', '8294')
    if not port.isdigit() or not 0 <= int(port) <= 65535:
        raise ValueError('invalid loopback port')
    return ['--build', str(BUILD), '--model-dir', '/model',
        '--manifest', '/inputs/model-manifest.json', '--manifest-sha256', digest,
        '--draft-environment', '/inputs/draft-environment.json',
        '--evidence', '/evidence/run-'+uuid.uuid4().hex, '--port', port]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve-seconds', type=float)
    args = parser.parse_args(argv)
    options = launch_arguments(os.environ)
    if args.serve_seconds is not None:
        if not 0 < args.serve_seconds <= 3600:
            raise ValueError('invalid bounded run duration')
        options += ['--serve-seconds', str(args.serve_seconds)]
    from augur_product import main
    main(options)


if __name__ == '__main__':
    main()

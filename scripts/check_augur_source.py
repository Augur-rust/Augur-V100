"""Run the shipped CPU checks without weights, CUDA, DSH or a live service."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import prepare_augur_release as R


def main():
    root = Path(__file__).resolve().parents[1]
    node = shutil.which('node')
    if not node:
        raise SystemExit('Node.js 22+ is required for the public JavaScript checks')
    tests = sorted(name[:-3] for name in R.PUBLIC_SCRIPTS
                   if name.startswith('test_') and name.endswith('.py'))
    missing = [name for name in tests if not (root / 'scripts' / (name + '.py')).is_file()]
    if missing:
        raise SystemExit('Missing public tests: ' + ', '.join(missing))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    steps = [
        ('Python public suite', [sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', *tests], root / 'scripts'),
        ('Browser chat regressions', [node, '--test', 'scripts/test_augur_web.cjs'], root),
        ('DSH single-flight regressions', [node, 'integrations/dsh/reasoning-recovery/single-flight.test.mjs'], root),
    ]
    for label, command, cwd in steps:
        print(label, flush=True)
        subprocess.run(command, cwd=cwd, env=env, check=True)
    for folder in ('scripts/augur_setup_web', 'scripts/augur_monitor_web'):
        for script in sorted((root / folder).glob('*.js')):
            subprocess.run([node, '--check', str(script)], cwd=root, env=env, check=True)
    print('PASS: public CPU checks; GPU, full DSH integration and clean-machine deployment not run.', flush=True)


if __name__ == '__main__':
    main()

"""Windows single-owner admission and bounded, credential-free startup diagnostics.

The OS mutex covers every port/package in this logon session and is held through
native retirement. Legacy/orphan native processes are detected without reading
command lines or killing them. CUDA errors are classified, never echoed verbatim.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import augur_launch_config as C


class EngineLease:
    def __init__(self, name=r'Local\Augur.NativeEngine.v1'):
        self.name, self.handle = name, None

    def __enter__(self):
        if os.name != 'nt':
            return self
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateMutexW.argtypes = [W.LPVOID, W.BOOL, W.LPCWSTR]
        k.CreateMutexW.restype = W.HANDLE
        k.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
        k.WaitForSingleObject.restype = W.DWORD
        k.ReleaseMutex.argtypes = [W.HANDLE]
        k.CloseHandle.argtypes = [W.HANDLE]
        self.kernel = k
        handle = k.CreateMutexW(None, False, self.name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        status = k.WaitForSingleObject(handle, 0)
        if status not in (0, 0x80):  # acquired or abandoned by a crashed product
            error = ctypes.get_last_error()
            k.CloseHandle(handle)
            if status == 0x102:
                raise RuntimeError('augur_engine_already_running: another Augur engine is starting or running; stop it in its own launcher before retrying')
            raise ctypes.WinError(error)
        self.handle = handle
        try:
            owners = native_owners()
            if owners:
                raise RuntimeError('augur_engine_already_running: native engine PID ' +
                    ', '.join(str(pid) for pid in owners) + '; stop it in its own launcher before retrying')
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        if self.handle is not None:
            self.kernel.ReleaseMutex(self.handle)
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def native_owners():
    """Read only PID/executable names; also reject pre-lease builds and orphans."""
    if os.name != 'nt':
        return []
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', W.DWORD), ('cntUsage', W.DWORD), ('th32ProcessID', W.DWORD),
            ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', W.DWORD),
            ('cntThreads', W.DWORD), ('th32ParentProcessID', W.DWORD),
            ('pcPriClassBase', W.LONG), ('dwFlags', W.DWORD), ('szExeFile', W.WCHAR * 260)]
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
    k.CreateToolhelp32Snapshot.restype = W.HANDLE
    k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [W.HANDLE, ctypes.POINTER(Entry)]
    k.Process32FirstW.restype = k.Process32NextW.restype = W.BOOL
    k.CloseHandle.argtypes = [W.HANDLE]
    handle = k.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    owners = []
    try:
        entry = Entry(); entry.dwSize = ctypes.sizeof(entry)
        ok = k.Process32FirstW(handle, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() in ('augur_session_jsonl.exe', 'augur_session_http.exe'):
                owners.append(int(entry.th32ProcessID))
            ok = k.Process32NextW(handle, ctypes.byref(entry))
        error = ctypes.get_last_error()
        if error != 18:  # ERROR_NO_MORE_FILES; never silently pass a failed enumeration
            raise ctypes.WinError(error)
        return sorted(owners)
    finally:
        k.CloseHandle(handle)


def tail(path, limit=16384):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read(limit).decode('utf8', 'replace')
    except OSError:
        return ''


def failure_detail(run_dir, exit_code=None):
    run_dir = Path(run_dir)
    logs = [run_dir / 'launcher.stderr.log', run_dir / 'launch/adapter.stderr.log']
    logs += list((run_dir / 'launch/native').glob('owner-*/stderr.log'))[-8:]
    for path in logs:
        text = tail(path)
        if 'augur_engine_already_running:' in text:
            return C.L('已有 Augur 引擎正在启动或运行（包括其他端口/版本）。请在原启动器中关闭服务后重试。',
                       'Another Augur engine is starting or running, possibly on another port/version. Stop it in its own launcher and retry.')
        if 'cuda_device_alloc' in text and 'raw_cuda_status: 2 }' in text:
            return C.L('CUDA 显存不足，模型未能加载。请先关闭仍占用显存的旧引擎或其他 GPU 程序，再重试。',
                       'CUDA ran out of GPU memory while loading the model. Stop the old engine or other GPU workloads, then retry.')
    suffix = '' if exit_code is None else C.L('，退出码 ', ', exit code ') + str(exit_code)
    return C.L('引擎在发送就绪回执前退出', 'The engine exited before sending its ready receipt') + suffix


def failed_start_retired(run_dir, result):
    """Process exit after failed initialization is distinct from verified CUDA cleanup.

Only a pre-ready failure can use this path. The caller separately waits for its
product process; a fresh lease checks for orphan adapters/producers on retry.
"""
    launch = Path(run_dir) / 'launch'
    if not launch.exists():  # validation/lease failed before any adapter was spawned
        return True
    retirements = result.get('native_retirements', [])
    shutdown = result.get('native_shutdown')
    if not isinstance(shutdown, dict) or not isinstance(retirements, list):
        return False
    return (not (launch / 'ready.json').exists() and result.get('confirmed_exit') is True
        and result.get('exit_code') == 0 and result.get('forced') is False
        and result.get('cleanup_errors') == []
        and shutdown.get('no_owner') is True
        and bool(retirements) and all(isinstance(r, dict) and r.get('confirmed_exit') is True
            and r.get('reason') == 'startup_failure' and r.get('force_attempted') is False
            for r in retirements))

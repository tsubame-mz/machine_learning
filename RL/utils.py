def _windows_enable_ANSI(std_id=1):
    # ANSIエスケープを有効にする0
    from ctypes import byref, POINTER, windll, WINFUNCTYPE
    from ctypes.wintypes import BOOL, DWORD, HANDLE

    GetStdHandle = WINFUNCTYPE(HANDLE, DWORD)(("GetStdHandle", windll.kernel32))
    GetFileType = WINFUNCTYPE(DWORD, HANDLE)(("GetFileType", windll.kernel32))
    GetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, POINTER(DWORD))(("GetConsoleMode", windll.kernel32))
    SetConsoleMode = WINFUNCTYPE(BOOL, HANDLE, DWORD)(("SetConsoleMode", windll.kernel32))

    if std_id == 1:  # stdout
        h = GetStdHandle(-11)
    elif std_id == 2:  # stderr
        h = GetStdHandle(-12)
    else:
        return False

    if h is None or h == HANDLE(-1):
        return False

    FILE_TYPE_CHAR = 0x0002
    if (GetFileType(h) & 3) != FILE_TYPE_CHAR:
        return False

    mode = DWORD()
    if not GetConsoleMode(h, byref(mode)):
        return False

    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    if (mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING) == 0:
        SetConsoleMode(h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    return True

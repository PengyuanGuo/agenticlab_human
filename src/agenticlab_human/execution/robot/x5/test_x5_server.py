"""Manual X5 API smoke test for the Server PC.

This file is named like a test, but it is a hardware diagnostic script. Keep
the xapi import inside main() so pytest can run on machines without the X5 SDK.
"""

from __future__ import annotations

import argparse


__test__ = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a direct xapi connection to X5.")
    parser.add_argument("--robot-ip", default="192.168.1.7")
    args = parser.parse_args()

    import xapi.api as x5

    x5.enable_debug_output(0)
    handle = x5.connect(args.robot_ip)
    print(f"\n 控制器句柄号：{handle}")
    try:
        version = x5.get_version(handle)
        print(f"版本信息：{version}")
        data = x5.get_do(handle, index=0)
        print(f"DO0 端口状态：{data}\n")
    finally:
        if handle is not None and handle != -1:
            x5.disconnect(handle)


if __name__ == "__main__":
    main()

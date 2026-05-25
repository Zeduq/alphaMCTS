# -*- coding: utf-8 -*-
import sys
import os


def check_python_version():
    print("[1/5] 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [FAIL] Python {version.major}.{version.minor}.{version.micro} (需要3.7+)")
        return False


def check_pyqt5():
    print("[2/5] 检查PyQt5...")
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QT_VERSION_STR
        print(f"  [OK] PyQt5 {QT_VERSION_STR}")
        return True
    except ImportError:
        print("  [FAIL] PyQt5 未安装")
        print("  -> 安装命令: pip install PyQt5")
        return False


def check_project_modules():
    print("[3/5] 检查项目模块...")
    modules = [
        'config',
        'search.tree',
        'search.lifecycle',
        'inference.agents.portrait',
        'inference.agents.formula',
        'evaluation.evaluator',
        'constraint.library',
        'constraint.fsa'
    ]

    missing = []
    for module in modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError as e:
            print(f"  [FAIL] {module}: {e}")
            missing.append(module)

    return len(missing) == 0


def check_data_file():
    print("[4/5] 检查数据文件...")
    data_path = 'D:/AAProject/Data/000300SH.csv'
    if os.path.exists(data_path):
        size = os.path.getsize(data_path) / (1024 * 1024)
        print(f"  [OK] 数据文件存在 ({size:.2f} MB)")
        return True
    else:
        print(f"  [FAIL] 数据文件不存在: {data_path}")
        return False


def check_api_config():
    print("[5/5] 检查API配置...")
    try:
        from config import OPENAI_API_KEY, BASE_URL
        if OPENAI_API_KEY and len(OPENAI_API_KEY) > 10:
            print(f"  [OK] API Key 已配置 ({OPENAI_API_KEY[:10]}...)")
            print(f"  [OK] Base URL: {BASE_URL[:30]}...")
            return True
        else:
            print("  [FAIL] API Key 未配置或无效")
            return False
    except Exception as e:
        print(f"  [FAIL] 配置检查失败: {e}")
        return False


def main():
    print("=" * 60)
    print("AlphaMCTS GUI 环境检查")
    print("=" * 60)
    print()

    checks = [
        ("Python版本", check_python_version),
        ("PyQt5", check_pyqt5),
        ("项目模块", check_project_modules),
        ("数据文件", check_data_file),
        ("API配置", check_api_config),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  错误: {e}")
            results.append((name, False))
        print()

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print("=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{name:<20} {status}")
    print()
    print(f"总计: {passed}/{total} 项检查通过")

    if passed == total:
        print("\n环境检查通过，可以运行 GUI 系统: python mcts_gui.py")
        return 0
    else:
        print("\n环境检查未完全通过，请根据提示修复问题。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

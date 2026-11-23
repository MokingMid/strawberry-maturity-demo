"""单元测试包。

存在的理由：tests/ 下各测试模块用 `from tests.helpers import ...` 绝对导入，
而交接文档给出的验证命令是 `python -m unittest discover -s tests -v`。
缺少本文件时 `tests` 不是可导入包，unittest 会把 start_dir 顶部目录直接塞进
sys.path，导致 `tests.helpers` 解析失败（5 个模块导入报错）。
加上本文件后 discover 会自动把项目根目录作为 top_level_dir，命令即可正常工作。
"""

from __future__ import annotations

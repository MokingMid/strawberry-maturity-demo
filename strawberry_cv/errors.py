"""统一异常类型。"""


class DependencyError(RuntimeError):
    """可选依赖未安装。"""


class PipelineError(RuntimeError):
    """推理/训练前置条件不满足。"""

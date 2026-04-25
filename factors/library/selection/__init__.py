"""内置选股因子库。

导入各子模块以触发 @register_factor 装饰器，将因子注册到 FactorRegistry。
"""

from . import momentum, volatility  # noqa: F401

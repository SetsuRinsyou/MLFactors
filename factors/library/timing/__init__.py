"""内置择时因子库。

导入各子模块以触发 @register_factor 装饰器，将因子注册到 FactorRegistry。
"""

from . import ma_cross, rsi  # noqa: F401

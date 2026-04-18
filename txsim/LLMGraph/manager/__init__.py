from LLMGraph.registry import Registry
manager_registry = Registry(name="ManagerRegistry")

# 导入必要的模块（其他模块有依赖）
# from .general import GeneralManager  # 注释掉：transaction 场景不需要，且依赖 retriever
from .transaction import TransactionManager

# 注释掉不需要的模块（避免导入错误）
# from .article import ArticleManager  # 依赖不存在的 loader.article
# from .movie import MovieManager  # 依赖不存在的 loader.movie
# from .social import SocialManager  # 不需要
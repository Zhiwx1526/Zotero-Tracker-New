import logging
import os
import sys

import dotenv
import hydra
from loguru import logger
from omegaconf import DictConfig

from zotero_tracker.executor import Executor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config: DictConfig) -> None:
    log_level = "DEBUG" if config.executor.debug else "INFO"
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )
    for name in logging.root.manager.loggerDict:
        if "zotero_tracker" in name:
            continue
        logging.getLogger(name).setLevel(logging.WARNING)

    if config.executor.debug:
        logger.info("已开启调试模式")

    Executor(config).run()


if __name__ == "__main__":
    main()

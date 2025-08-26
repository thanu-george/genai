import traceback
from logging import DEBUG, Logger

def handle_milvus_exception(e: Exception, logger: Logger):
    if logger.isEnabledFor(DEBUG):
        # Full traceback for DEBUG
        logger.debug(f"Milvus Exception: {str(e)}", exc_info=True)
    else:
        # Short message + last traceback line for INFO
        tb = traceback.extract_tb(e.__traceback__)
        last_frame = tb[-1] if tb else None
        logger.info(
            f"Milvus Error: {str(e)} "
            f"(File: {last_frame.filename if last_frame else '?'}, "
            f"Line: {last_frame.lineno if last_frame else '?'})"
        )

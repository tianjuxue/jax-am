import logging

# Define the global logger - by default set to INFO level
logger = logging.getLogger()
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("rocm").setLevel(logging.ERROR)
logging.getLogger("tpu").setLevel(logging.ERROR)

logger.setLevel(logging.INFO)

# Define your formatter
formatter = logging.Formatter('[%(asctime)s - %(levelname)s] - %(message)s')

# Define and add the stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

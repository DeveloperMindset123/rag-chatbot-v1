# TODO : delete later, used for experimental purpose only
import logging

logger = logging.getLogger(__name__)


def do_something():
    logger.info("doing something")
    logger.debug("debugging something")


def main():
    logging.basicConfig(filename="myApp.log", level=logging.INFO)
    logging.info("Started")
    do_something()
    logging.info("Finished")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
from TestDataHolder import TestDataHolder
from TestTrainer import TestTrainer

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from util import run_test  # noqa


def main():
    run_test(TestDataHolder)
    run_test(TestTrainer)


if __name__ == "__main__":
    main()

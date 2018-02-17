#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import inspect
from TestDataMani import TestDataMani
from TestDataAug import TestDataAug

almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from util import run_test # noqa


def main():
    run_test(TestDataMani)
    run_test(TestDataAug)


if __name__ == "__main__":
    main()

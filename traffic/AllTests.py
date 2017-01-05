import unittest

from traffic.data_explorer_test import *

def suite():
    suite = unittest.TestSuite()
    suite.addTests([TestSignNames(), TestDataExplorer(), TestTrainingPlotter()])
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)

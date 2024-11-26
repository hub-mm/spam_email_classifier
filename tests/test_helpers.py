# ./tests/test_helpers.py
import unittest
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.data_labelling import is_spam

class TestDataLabelling(unittest.TestCase):
    def test_is_spam_true(self):
        email = "Congratulations! You have won a free lottery. Click here to claim."
        self.assertTrue(is_spam(email))

    def test_is_spam_false(self):
        email = "Dear friend, let's catch up over coffee tomorrow."
        self.assertFalse(is_spam(email))

if __name__ == '__main__':
    unittest.main()
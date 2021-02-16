# -*- coding: utf-8 -*-
import benchmarks_cnn_api.models.deep_api as deep_api
import collections
import unittest

class TestModelMethods(unittest.TestCase):
    
    def setUp(self):
        self.meta = deep_api.get_metadata()
        
    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)
        
    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(self.meta['Name'].replace('-','').replace('_',''),
                        'benchmarks_cnn_api'.replace('-','').replace('_',''))
        self.assertEqual(self.meta['Author'], 'A.Grupp, V.Kozlov (KIT)')
        self.assertEqual(self.meta['Author-email'], 'valentin.kozlov@kit.edu')

    def test_model_get_train_args(self):
        """
        Test that get_train_args returns dict
        """
        # get_train_args() may return <class 'collections.OrderedDict'>
        # to simplify the test, we assume dict => use "update()"        
        train_args = {}
        train_args.update(deep_api.get_train_args())
        print("TRAIN_ARGS:", train_args, type(train_args))
        self.assertTrue(type(train_args) is dict)
        # in the case of OrderedDict may use the following:
        # train_args = deep_api.get_train_args()
        # self.assertTrue(type(train_args) is collections.OrderedDict)

if __name__ == '__main__':
    unittest.main()

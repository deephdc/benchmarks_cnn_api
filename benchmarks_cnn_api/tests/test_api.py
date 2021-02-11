# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under its License. Please, see the LICENSE file
#
"""
Created on Thu Feb 11 19:51:44 2021

@author: vykozlov
"""

from aiohttp import test_utils
from aiohttp import web
import deepaas.model.v2
import json
import sys
import unittest

from deepaas.api import v2
from deepaas import config
from deepaas.tests import base
from deepaas.model import loading


class TestApiV2(base.TestCase):
    async def get_application(self):

        app = web.Application(debug=True)
        app.middlewares.append(web.normalize_path_middleware())

        deepaas.model.v2.register_models(app)

        v2app = v2.get_app()
        app.add_subapp("/v2", v2app)

        return app

    def setUp(self):
        super(TestApiV2, self).setUp()
        self.flags(debug=True)

    def assert_ok(self, response):
        self.assertIn(response.status, [200, 201])

    # the unittest_run_loop decorator can be used in tandem with
    # the AioHTTPTestCase to simplify running
    # tests that are asynchronous
    @test_utils.unittest_run_loop
    async def test_versions_returns_200(self):
        resp = await self.client.get("/v2/")
        self.assert_ok(resp)

    @test_utils.unittest_run_loop
    async def test_models_returns_200(self):
        resp = await self.client.get("/v2/models/")
        self.assert_ok(resp)

    @test_utils.unittest_run_loop
    async def test_models_model_returns_200(self):
        resp = await self.client.get("/v2/models/benchmarks_cnn/")
        self.assert_ok(resp)
        
    @test_utils.unittest_run_loop
    async def test_models_model_get_train_returns_200(self):
        resp = await self.client.get("/v2/models/benchmarks_cnn/train/")
        self.assert_ok(resp)
        #self.assertTrue(type(resp.text()) is dict)

    @unittest.skip("Spurious failure due to to asyncio closed loop")
    @test_utils.unittest_run_loop
    async def test_models_model_post_train_returns_200(self):
        ret = await self.client.post("/v2/models/benchmarks_cnn/train/",
                                     data={})
        self.assertEqual(200, ret.status)


if __name__ == '__main__':
    config.config_and_logging(sys.argv) # why this is SO important??
    unittest.main()

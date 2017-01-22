#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from collections import OrderedDict
dict=OrderedDict

import pandas
from pandas.testing import assert_frame_equal, assert_series_equal
from Chassis import Chassis

class SimpleTests(unittest.TestCase):
	def setUp(self):
		ds=pandas.DataFrame.from_records([
			{"c": "A", "n":10, "b": True, "s":1, "w": 1},
			{"c": "B", "n":11, "b": False, "s":1, "w": 10},
			{"c": "C", "n":20, "b": 0.5, "s":1, "w": 5},
			{"c": "A", "n":42, "b": 0.3, "s":1, "w": 6},
		]).to_dense()
		ds.loc[:,"c"]=pandas.Series(pandas.Categorical( ds.loc[:,"c"]))
		
		self.schema={
			"c": "categorical",
			"n": "numerical",
			"b": "binary",
			"w": "weight",
			"s": "stop", #this won't appear in the dmat
		}
		self.chs=Chassis(self.schema, ds)
		b=ds.loc[:,"b"]
		n=ds.loc[:,"n"]
		abc=pandas.concat([ds.loc[:,"c"]=="A", ds.loc[:,"c"]=="B", ds.loc[:,"c"]=="C"], axis=1, ).astype(int)
		abc.columns=("c_A", "c_B", "c_C")
		
		self.expected={
			'c' : (ds.loc[:,["b", "n"]], ds["c"]),
			'n' : (pandas.concat([abc, b], axis=1), n),
			'b' : (pandas.concat([abc, n], axis=1), b)
		}
		self.ds=ds
	
	def shouldColumnBePresent(self, cn):
		return self.schema[cn] not in {"stop", "weight"}
	
	def checkCovariates(self, cn):
		if self.shouldColumnBePresent(cn):
			assert_frame_equal(self.chs.prepareCovariates(cn), self.expected[cn][0], check_like=True, check_dtype=False)
	
	def checkResult(self, cn):
		if self.shouldColumnBePresent(cn):
			assert_series_equal(self.chs.prepareResults(cn), self.expected[cn][1], check_dtype=False, check_categorical=False)
		else:
			with self.assertRaises(Exception):
				self.chs.prepareResults(cn)
	
	def testCovariates(self):
		for cn in self.schema:
			self.checkCovariates(cn)
	
	def testResult(self):
		for cn in self.schema:
			self.checkResult(cn)


if __name__ == '__main__':
	unittest.main()

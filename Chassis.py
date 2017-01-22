__all__ = ("Chassis", "MissingColumnsExplainer", "resampleDataFrame")


import typing
import math

try:
	from math import nan
except:
	nan = float("nan")

from collections import OrderedDict
import numpy as np
import pandas
import warnings
from functools import wraps


class RecomputingDict(dict):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def mutateCallback(self):
		pass

	def __setitem__(self, *args, **kwargs):
		super().__setitem__(*args, **kwargs)


#_datasetsDb:typing.Optional[typing.Mapping[str, typing.Callable[[], object]]]=None
_datasetsDb = None


def _checkAndInitDatasetsDb() -> None:
	"""creates a dict mapping datasets names to the functions loading them"""
	global _datasetsDb
	if _datasetsDb is None:
		import sklearn.datasets
		import re

		dsLoadFuncNameRx = re.compile("^(load|fetch)_(.+)")
		_datasetsDb = {}
		for fName in dir(sklearn.datasets):
			m = dsLoadFuncNameRx.match(fName)
			if m:
				_datasetsDb[m.group(2)] = getattr(sklearn.datasets, fName)


def resampleDataFrame(pdf: pandas.DataFrame, balancer, columns: typing.Set[str] = None) -> pandas.DataFrame:
	"""Use a resampler from imblearn to resample dataframe columns"""
	origCols = pdf.columns
	if columns is None:
		columns = origCols
	for cn in columns:
		x = pdf.loc[:, list(set(origCols) - {cn})]
		y = pdf.loc[:, cn]
		x1, y1 = balancer.fit_sample(x, y)
		x1 = pandas.DataFrame(x1, columns=x.columns)
		y1 = pandas.Series(y1, name=y.name)
		pdf = pandas.concat([x1, y1], axis=1)
	pdf.reindex(origCols, axis=1)
	return pdf


class MissingColumnsExplainer:
	"""If a column is missing in covariates matrix (the one with encoded cathegoricals and without `stop` columns), tries to find an explaination and fix it"""

	__slots__ = ("parent", "categorical", "unexplained")

	def __init__(self, parent, missingColumns):
		self.unexplained = set(missingColumns)
		self.categorical = []
		for mcn in missingColumns:
			colIsLikelyCategorical = False
			for ccn in parent.groups["categorical"]:
				if mcn.startswith(ccn):
					self.unexplained.remove(mcn)
					self.categorical.append(mcn)
					break

	def fix(self, dmat, raiseUnexplained=True, warnFixable=False):
		if self.categorical and warnFixable:
			warnings.warn("the design matrix resulted from the dataset has no " + repr(self.categorical) + " columns, but they are present in the model. The cause are likely categorical variables values one-hot encoded missing in your partition of the dataset, so creating their columns with 0.-filled")

			for mcn in self.categorical:
				dmat.loc[:, mcn] = 0.0

		if self.unexplained and raiseUnexplained:
			raise ValueError("Columns `" + repr(self.unexplained) + "` are missing from the design matrix and we cannot explain this.")


StrOrStrIter = typing.Union[str, typing.Iterable[str]]

from inspect import signature
def allowAcceptMultipleColumns(fOrSeriesClass=None, *, seriesClass:type=None):
	if isinstance(fOrSeriesClass, type):
		seriesClass = fOrSeriesClass
		f = None
	elif callable(fOrSeriesClass):
		f = fOrSeriesClass
	else:
		raise ValueError()
	
	def _allowAcceptMultipleColumns(f):
		s = signature(f)
		firstParam = list(s.parameters.values())[1]
		assert firstParam.name == "cn", firstParam.name
		assert firstParam.annotation == str, firstParam.annotation
		assert s.return_annotation == np.ndarray or s.return_annotation == pandas.Series or s.return_annotation == pandas.DataFrame, s.return_annotation
		
		if seriesClass is None or seriesClass is s.return_annotation:
			singleSeriesCall = f
			seriesClass_ = s.return_annotation
		else:
			seriesClass_ = seriesClass
			def singleSeriesCall(self, cn: str, *args, **kwargs) -> seriesClass_:
				return seriesClass(f(self, cn, *args, **kwargs))
		
		@wraps(f)
		def modifiedF(self, cns: StrOrStrIter, *args, **kwargs) -> typing.Union[pandas.DataFrame, seriesClass_]:
			if isinstance(cns, str):
				return singleSeriesCall(self, cns, *args, **kwargs)
			else:
				return pandas.concat((f(self, cn, *args, **kwargs) for cn in cns), axis=1)
		
		#modifiedF.__name__ = f.__name__.replace("col", "cols")
		modifiedF.__name__ = f.__name__
		modifiedF.__annotations__["cn"] = StrOrStrIter
		return modifiedF
	
	if f is not None:
		return _allowAcceptMultipleColumns(f)
	else:
		return _allowAcceptMultipleColumns

class Chassis:
	"""Patsy is shit. This class prepares data, and it's more predictable than patsy"""

	__slots__ = ("columns", "groups", "features", "stop", "weights", "catIndex", "catRemap", "pds", "dontWarnAboutMissingStopColumns")

	#columns:typing.Set[str]
	groupsTypes = {gn: gn for gn in ("categorical", "numerical", "stop", "weight", "binary")}

	def __init__(self, spec: typing.Mapping[str, str], dataset: typing.Optional[pandas.DataFrame] = None) -> None:
		"""Imports `dataset` according to the `spec`
		`spec` is a dict specifying schema of your data. Its keys are columns names, its values are strings "Categoric", "Numeric", "Binary" and "Stop" (it takes into account only first letters). "Stop" are removed.
		`dataset` is a `pandas.DataFrame` with your data.
		"""
		if isinstance(dataset, __class__):
			if spec is None:
				spec = dataset.spec
			dataset = dataset.pds
		
		self.dontWarnAboutMissingStopColumns = False
		self.weights = None
		self.importSpec(spec)
		self.importDataset(dataset)

	def _reprContents(self) -> str:
		return ", ".join(("columns: " + str(len(self.columns)), ", ".join(((gn + ": " + str(len(g))) for gn, g in self.groups.items() if len(g)))))

	def __repr__(self) -> str:
		return self.__class__.__name__ + "< " + self._reprContents() + " >"

	def importSpec(self, spec: typing.Mapping[str, str]) -> None:
		"""Imports specification dictionary."""
		self.features = spec
		self.groups = {gtn: set() for gtn in set(self.__class__.groupsTypes.values())}  # slow, need to move into metaclass
		for k, c in self.features.items():
			self.groups[self.__class__.groupsTypes[c.lower()]].add(k)
		self.columns = set(self.features) - set(self.groups["stop"]) - set(self.groups["weight"])

	#catIndex: typing.Mapping[str, pandas.Series]
	#catRemap: typing.Mapping[str, ]

	def importDataset(self, pds: pandas.DataFrame) -> None:
		"""Transforms pandas.DataFrame `pds` into internal representation."""
		if pds is None:
			self.pds = None
			self.catIndex = {}
			self.catRemap = {}
			return

		pds.reindex()
		if hasattr(pds, "infer_objects"):
			pds = pds.infer_objects()

		presentStopColumns = self.groups["stop"] & set(pds.columns)
		missingStopColumns = self.groups["stop"] - presentStopColumns

		if missingStopColumns and not self.dontWarnAboutMissingStopColumns:
			warnings.warn("Following stop columns are missing: " + repr(missingStopColumns) + ". Using only present columns.")
			self.columns -= missingStopColumns
			self.groups["stop"] = presentStopColumns

		self.groups["weight"] = self.groups["weight"] & set(pds.columns)
		if self.groups["weight"]:
			assert len(self.groups["weight"]) == 1
			self.weights = pds.loc[:, list(self.groups["weight"])]

		self.stop = pds.loc[:, list(presentStopColumns)]
		
		#print(self.)
		pds = pds.loc[:, list(self.columns)]

		colz = [pds[cn].astype("float32") for cn in self.groups["binary"]]
		catColz = {cn: pds[cn].astype("category") for cn in self.groups["categorical"]}
		dummiez = {cn: pandas.get_dummies(col, prefix=cn) for cn, col in catColz.items()}
		self.catIndex = {cn: col.cat.categories for cn, col in catColz.items()}
		self.catRemap = {cn: list(col.columns) for cn, col in dummiez.items()}
		#for cn in catColz:
		#	print(cn, len(self.catIndex[cn]), len(self.catRemap[cn]))
		#	assert(len(self.catIndex[cn])==len(self.catRemap[cn]))
		colz.extend(dummiez.values())
		colz.extend([pandas.to_numeric(pds[c], "coerce") for c in self.groups["numerical"]])
		
		self.pds = pandas.concat(colz, axis=1)

	@allowAcceptMultipleColumns(pandas.DataFrame)
	def _colsNaEquiv(self, cn: str) -> pandas.Series:
		"""Returns a column suitable for checking if a value is nan. If it is categorical, it selects the first column one-hot because if value is nan one-hot will make all the values nans"""
		if cn in self.catRemap:
			col = self.pds.loc[:, self.catRemap[cn][0]]
		else:
			col = self.pds.loc[:, cn]
		return col


	def colsNotNA(self, cns: StrOrStrIter) -> pandas.Series:
		"""Returns result of comparison of original column values to nans"""
		return self._colsNaEquiv(cns).notna().all(axis=1, skipna=False)

	def colsIsNA(self, cns: StrOrStrIter) -> pandas.Series:
		"""Returns result of comparison of original column values to nans"""
		return self._colsNaEquiv(cns).isna().any(axis=1, skipna=False)

	def prepareCovariates(self, cns: typing.Optional[StrOrStrIter] = (), dmat: typing.Optional[pandas.DataFrame] = None, excludeColumns:typing.Set[str] = None) -> pandas.DataFrame:
		"""Returns matrix of the rest of covariates needed to fit column `cn`"""
		if dmat is None:
			dmat = self.pds
		neededCols = set(dmat.columns)
		if excludeColumns is not None:
			neededCols -= excludeColumns

		if cns is None:
			cns = tuple()
		elif isinstance(cns, str):
			cns = (cns,)

		for cn in cns:
			if cn in self.catRemap:
				neededCols -= set(self.catRemap[cn])
			else:
				neededCols -= {cn}
				#print(neededCols)
		
		return dmat.loc[:, list(neededCols)]

	def oneHotToCategory(self, cn: str, oneHot: pandas.DataFrame, index=None) -> pandas.Series:
		"""Reverses one-hot encoding for category name. Transforms a matrix of columns `oneHot` (it must contain ONLY that columns, AND in the right order) into a column with type `category`"""
		#print(cn, self.catIndex, self.catRemap)
		#print(cn, len(self.catIndex[cn]), len(self.catRemap[cn]))
		assert len(self.catIndex[cn]) == len(self.catRemap[cn])
		catIdx = self.catIndex[cn]
		return self.numpyToColumn(cn, pandas.Categorical(catIdx[np.argmax(oneHot, axis=1)], categories=catIdx, ordered=False), index)  # TODO: NaN = null vec

	def numpyToColumn(self, cn: str, data: np.array, index=None) -> pandas.Series:
		"""Converts a numpy array into a column"""
		if index is None:
			index = self.pds.index
		res = pandas.pandas.Series(data, index=index)
		res.name = cn
		return res

	def decodeCategory(self, cn: str, dmat: typing.Optional[pandas.DataFrame] = None) -> pandas.Series:
		"""Returns original (like in the inital pandas.DataFrame) representation of column with the name `cn`."""
		if dmat is None:
			dmat = self.pds
		return self.oneHotToCategory(
			cn, np.array(
				dmat.loc[:, list(self.catRemap[cn])]
			)
		)

	@allowAcceptMultipleColumns
	def prepareResults(self, cn: str, dmat: typing.Optional[pandas.DataFrame] = None) -> pandas.Series:
		"""Prepares result column pandas.DataFrame"""
		if dmat is None:
			dmat = self.pds
		if cn in self.catRemap:
			return self.decodeCategory(cn, dmat)
		else:
			return dmat.loc[:, cn]

	def select(self, decodeCategories: bool = True, columns: typing.Optional[typing.Set[str]] = None):
		"""Returns matrix by original columns, not transformed ones.
		decodeCategories transforms one-hot encoded columns back to the original ones
		columns allows to select subset of columns. If it is None, all the original columns are selected.
		"""
		if columns is None:
			columns = self.columns | self.groups["stop"]

		colz = [self.pds.loc[:, cn] for cn in (self.groups["binary"] & columns)]
		colz.extend([self.pds.loc[:, cn] for cn in (self.groups["numerical"] & columns)])
		colz.append(self.stop[list(columns & self.groups["stop"])])
		if self.weights is not None:
			colz.append(self.weights)
		res = pandas.concat(colz, axis=1)
		if decodeCategories:
			for cn in set(self.catRemap) & columns:
				res[cn] = self.decodeCategory(cn)
		else:
			for cn in set(self.catRemap) & columns:
				for vcn in self.catRemap[cn]:
					res[vcn] = self.pds.loc[:, vcn]
		return res

	def reduceCategoricalCols(self, dmat: typing.Optional[pandas.DataFrame], columns: typing.Optional[typing.Set[str]] = None):
		"""Sums categorical columns. In future may use other functions. Useful for combining additive values like SHAP scores."""
		if columns is None:
			columns = set(self.catRemap)

		availCols = set(dmat.columns)
		plainColumns = list((self.columns - columns) & availCols)

		resCols = [dmat.loc[:, list(plainColumns)]]
		for cn in columns:
			colz = dmat.loc[:, list(set(self.catRemap[cn]) & availCols)]
			colz = colz[colz.notna()]
			resC = colz.sum(axis=1)
			resC.name = cn
			resCols.append(resC)
		return pandas.concat(resCols, axis=1)

	def reverse(self, columns: typing.Optional[typing.Set[str]] = None):
		"""Encodes design matrix back into initial representation, can return subset of the original columns"""
		return self.select(decodeCategories=True, columns=columns)

	@staticmethod
	def specFromPandas(ds: pandas.DataFrame) -> typing.Mapping[str, str]:
		"""Tries to reverse-engineer spec from data."""
		spec = {}
		for cn in ds.columns:
			dt = ds.dtypes[cn]
			v = ds.loc[:, cn]
			rT = None
			if dt.kind is "f" or dt.kind is "i":
				if v[0] == 0 or v[0] == 1:
					tf = set(dt.type([True, False]))
					if tf == (set(v.unique()) & tf):
						rT = "binary"
					else:
						rT = "numerical"
				else:
					rT = "numerical"
			elif dt.kind is "O":
				types = set(v.map(type)) - {None.__class__}
				if len(types) == 1:
					tp = next(iter(types))
					if np.issubdtype(tp, str):
						rT = "categorical"
			if rT is None:
				rT = "stop"
			spec[cn] = rT
		return spec

	@classmethod
	def fromSKLearnDataset(cls, dataset: typing.Union[str, "sklearn.utils.Bunch"], targetName: str = "target", *args, **kwargs):
		"""Converts an sklearn dataset into a Chassis"""
		if isinstance(dataset, str):
			_checkAndInitDatasetsDb()
			dataset = _datasetsDb[dataset]()
		ds = pandas.concat([pandas.Series(dataset.target, name=targetName), pandas.DataFrame(dataset.data, columns=dataset.feature_names)], axis=1)
		ds = ds.infer_objects()
		spec = cls.specFromPandas(ds)
		return cls(spec, ds, *args, **kwargs)
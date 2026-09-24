"""Model pipelines and search spaces for the project."""

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import get_preprocessors


def _build_voting_classifier(class_weight):
	"""Build a soft-voting classifier with one weighting policy."""
	return VotingClassifier(
		estimators=[
			(
				"logistic_regression",
				LogisticRegression(
					class_weight=class_weight,
					max_iter=2000,
					random_state=42,
				),
			),
			(
				"random_forest",
				RandomForestClassifier(
					class_weight=class_weight,
					n_estimators=300,
					random_state=42,
					n_jobs=-1,
				),
			),
			(
				"svm",
				SVC(
					class_weight=class_weight,
					probability=True,
					random_state=42,
				),
			),
		],
		voting="soft",
		n_jobs=-1,
	)


def get_model_pipelines(X):
	"""Return normal and class-balanced pipelines for each model family.

	The returned dictionary contains ten independent pipelines: one normal and
	one class-balanced version of each of the five requested model families.
	Every pipeline owns its preprocessor, so cross-validation learns imputation,
	encoding, and scaling only from each training fold.
	"""
	preprocessor_scaled, preprocessor_unscaled = get_preprocessors(X)

	model_definitions = {
		"decision_tree": (
			preprocessor_unscaled,
			DecisionTreeClassifier(random_state=42),
			DecisionTreeClassifier(class_weight="balanced", random_state=42),
		),
		"logistic_regression": (
			preprocessor_scaled,
			LogisticRegression(max_iter=2000, random_state=42),
			LogisticRegression(
				class_weight="balanced",
				max_iter=2000,
				random_state=42,
			),
		),
		"random_forest": (
			preprocessor_unscaled,
			RandomForestClassifier(
				n_estimators=300,
				random_state=42,
				n_jobs=-1,
			),
			RandomForestClassifier(
				class_weight="balanced",
				n_estimators=300,
				random_state=42,
				n_jobs=-1,
			),
		),
		"svm": (
			preprocessor_scaled,
			SVC(probability=True, random_state=42),
			SVC(class_weight="balanced", probability=True, random_state=42),
		),
		"voting_ensemble": (
			preprocessor_scaled,
			_build_voting_classifier(None),
			_build_voting_classifier("balanced"),
		),
	}

	pipelines = {}
	for model_name, (preprocessor, normal_model, balanced_model) in model_definitions.items():
		pipelines[f"{model_name}_normal"] = Pipeline(
			steps=[("preprocessor", clone(preprocessor)), ("model", normal_model)]
		)
		pipelines[f"{model_name}_balanced"] = Pipeline(
			steps=[("preprocessor", clone(preprocessor)), ("model", balanced_model)]
		)

	return pipelines


#def get_param_grids():
	"""Return optional hyperparameter grids keyed by pipeline name.

	These grids can be passed to ``GridSearchCV`` or ``RandomizedSearchCV``.
	The balanced and normal versions are tuned separately, preserving the
	deliberate comparison between their class-weighting strategies.
	"""
	common_tree_grid = {
		"model__max_depth": [None, 5, 10, 20],
		"model__min_samples_leaf": [1, 5, 10],
	}
	common_linear_grid = {
		"model__C": [0.1, 1.0, 10.0],
	}

	grids = {
		"decision_tree": common_tree_grid,
		"logistic_regression": common_linear_grid,
		"random_forest": {
			"model__max_depth": [None, 10, 20],
			"model__min_samples_leaf": [1, 5],
			"model__max_features": ["sqrt", "log2"],
		},
		"svm": {
			"model__C": [0.1, 1.0, 10.0],
			"model__gamma": ["scale", "auto"],
		},
		"voting_ensemble": {
			"model__weights": [None, [2, 1, 1], [1, 2, 1], [1, 1, 2]],
		},
	}

	return {
		f"{model_name}_{variant}": grid
		for model_name, grid in grids.items()
		for variant in ("normal", "balanced")
	}

'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Date: July 5st, 2024
# Permissions and Citation: Refer to the README file.
'''

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

import optuna, random, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.ensemble import *
from sklearn.feature_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from xgboost import *

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)


# Ignore warnings.
def warn(*args, **kwargs):
  pass


def CalculateAllMetricsUpdated(cm):
  # UPDATE: To add the epsilon value to avoid division by zero.
  epsilon = 1e-7

  # Calculate TP, TN, FP, FN.
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  TN = np.sum(cm) - (TP + FP + FN)

  results = {}

  # Using macro averaging.
  precision = np.mean(TP / (TP + FP + epsilon))
  recall = np.mean(TP / (TP + FN + epsilon))
  f1 = 2 * precision * recall / (precision + recall + epsilon)
  accuracy = np.sum(TP) / (np.sum(cm) + epsilon)
  specificity = np.mean(TN / (TN + FP + epsilon))
  bac = (recall + specificity) / 2.0
  avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0

  results["Macro Precision"] = precision
  results["Macro Recall"] = recall
  results["Macro F1"] = f1
  results["Macro Accuracy"] = accuracy
  results["Macro Specificity"] = specificity
  results["Macro BAC"] = bac
  results["Macro AVG"] = avg

  # Using micro averaging.
  precision = np.sum(TP) / np.sum(TP + FP + epsilon)
  recall = np.sum(TP) / np.sum(TP + FN + epsilon)
  f1 = 2 * precision * recall / (precision + recall + epsilon)
  accuracy = np.sum(TP) / (np.sum(cm) + epsilon)
  specificity = np.sum(TN) / np.sum(TN + FP + epsilon)
  bac = (recall + specificity) / 2.0
  avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0

  results["Micro Precision"] = precision
  results["Micro Recall"] = recall
  results["Micro F1"] = f1
  results["Micro Accuracy"] = accuracy
  results["Micro Specificity"] = specificity
  results["Micro BAC"] = bac
  results["Micro AVG"] = avg

  # Using weighted averaging.
  samples = np.sum(cm, axis=1)
  weights = samples / np.sum(cm)

  precision = np.sum(TP / (TP + FP) * weights)
  recall = np.sum(TP / (TP + FN) * weights)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.sum(TN / (TN + FP) * weights)
  bac = (recall + specificity) / 2.0
  avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0

  results["Weighted Precision"] = precision
  results["Weighted Recall"] = recall
  results["Weighted F1"] = f1
  results["Weighted Accuracy"] = accuracy
  results["Weighted Specificity"] = specificity
  results["Weighted BAC"] = bac
  results["Weighted AVG"] = avg

  return results


def CalculateAllMetricsBinaryUpdated(cm):
  # UPDATE: To add the epsilon value to avoid division by zero.
  epsilon = 1e-7

  # Calculate TP, TN, FP, FN.
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  TN = np.sum(cm) - (TP + FP + FN)

  TN = TN[0]
  TP = TP[0]
  FP = FP[0]
  FN = FN[0]

  precision = TP / (TP + FP + epsilon)
  recall = TP / (TP + FN + epsilon)
  f1 = 2 * precision * recall / (precision + recall + epsilon)
  accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
  specificity = TN / (TN + FP + epsilon)
  bac = (recall + specificity) / 2.0
  avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0

  results = {
    "TP"         : TP,
    "TN"         : TN,
    "FP"         : FP,
    "FN"         : FN,
    "Precision"  : precision,
    "Recall"     : recall,
    "F1"         : f1,
    "Accuracy"   : accuracy,
    "Specificity": specificity,
    "BAC"        : bac,
    "AVG"        : avg,
  }

  return results


def LoadDataset(filePath, fillValue=0):
  # Load the dataset.
  data = pd.read_csv(filePath)
  # Shuffle the dataset.
  data = data.sample(frac=1).reset_index(drop=True)
  # Fill the missing values.
  data = data.fillna(fillValue)
  # Get the features and target columns.
  featuresColumns = data.columns[:-1]
  targetColumn = data.columns[-1]
  # Split the dataset into features and target.
  X = data[featuresColumns]
  y = data[targetColumn]
  # Print the dataset information.
  print("Features and Target Columns:")
  print(featuresColumns)
  print(targetColumn)
  print("Features and Target Shapes:")
  print(X.shape)
  print(y.shape)
  print("Target Value Counts:", y.value_counts())
  return X, y, featuresColumns, targetColumn


def EncodeData(y, technique, transformOnly=False):
  if (transformOnly):
    # Transform the data.
    y = technique.transform(y)
    encodeObj = technique
  else:
    # Create the encoding object.
    encodeObj = technique()
    # Fit and transform the data.
    y = encodeObj.fit_transform(y)
  return y, encodeObj


def NormalizeData(xTrain, xTest, techniqueStr):
  techniques = {
    "Normalize_L1_0" : lambda x: normalize(x, norm="l1", axis=0),
    "Normalize_L2_0" : lambda x: normalize(x, norm="l2", axis=0),
    "Normalize_Max_0": lambda x: normalize(x, norm="max", axis=0),
    "Normalize_L1_1" : lambda x: normalize(x, norm="l1", axis=1),
    "Normalize_L2_1" : lambda x: normalize(x, norm="l2", axis=1),
    "Normalize_Max_1": lambda x: normalize(x, norm="max", axis=1),
    # "Normalizer"     : Normalizer(),
    "StandardScaler" : StandardScaler(),
    "MinMaxScaler"   : MinMaxScaler(),
    "MaxAbsScaler"   : MaxAbsScaler(),
    "RobustScaler"   : RobustScaler(),
  }

  if (techniqueStr is None):
    return xTrain, xTest, None

  if (techniqueStr.startswith("Normalize")):
    xTrain = techniques[techniqueStr](xTrain)
    xTest = techniques[techniqueStr](xTest)
    return xTrain, xTest, None
  else:
    # Create the normalization object.
    normObj = techniques[techniqueStr]
    # Fit and transform the training data.
    xTrain = normObj.fit_transform(xTrain)
    # Transform the test data.
    xTest = normObj.transform(xTest)
    return xTrain, xTest, normObj


def TrainEvaluateModel(modelStr, hyperparams, xTrain, yTrain, xTest, yTest):
  models = {
    "RandomForestClassifier"    : RandomForestClassifier(),
    "DecisionTreeClassifier"    : DecisionTreeClassifier(),
    "SVC"                       : SVC(),
    "KNeighborsClassifier"      : KNeighborsClassifier(),
    "MLPClassifier"             : MLPClassifier(),
    "AdaBoostClassifier"        : AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "ExtraTreesClassifier"      : ExtraTreesClassifier(),
    "BaggingClassifier"         : BaggingClassifier(),
    "XGBClassifier"             : XGBClassifier(),
  }
  model = models[modelStr]  # Get the model.
  model.set_params(**hyperparams)  # Set the hyperparameters.
  model.fit(xTrain, yTrain)  # Train the model.
  predTest = model.predict(xTest)  # Evaluate the model.
  cm = confusion_matrix(yTest, predTest)  # Calculate the confusion matrix.
  if (len(np.unique(yTest)) == 2):
    metrics = CalculateAllMetricsBinaryUpdated(cm)  # Calculate the metrics.
  else:
    metrics = CalculateAllMetricsUpdated(cm)  # Calculate the metrics.
  return metrics, cm


def FeaturesSelection(xTrain, xTest):
  xLength = xTrain.shape[1]
  binIndices = np.random.choice([0, 1], size=xLength)
  if (np.sum(binIndices) == 0):
    binIndices[np.random.randint(0, xLength)] = 1
  xTrain = xTrain[:, binIndices]
  xTest = xTest[:, binIndices]
  return xTrain, xTest, list(binIndices)


def SampleDataset(xTrain, yTrain, techniqueStr):
  techniques = {
    "SMOTE"             : SMOTE,
    "ADASYN"            : ADASYN,
    "RandomOverSampler" : RandomOverSampler,
    "RandomUnderSampler": RandomUnderSampler,
    "NearMiss"          : NearMiss,
    "TomekLinks"        : TomekLinks,
    "SMOTETomek"        : SMOTETomek,
    "SMOTEENN"          : SMOTEENN,
    "SVMSMOTE"          : SVMSMOTE,
    "KMeansSMOTE"       : KMeansSMOTE,
    "BorderlineSMOTE"   : BorderlineSMOTE,
  }

  if (techniqueStr is None):
    return xTrain, yTrain, None

  # Create the sampling object.
  sampleObj = techniques[techniqueStr]()
  # Fit and transform the data.
  xTrain, yTrain = sampleObj.fit_resample(xTrain, yTrain)
  return xTrain, yTrain, sampleObj


def OptunaObjectiveFunction(
  trial,
  xTrain,
  yTrain,
  xTest,
  yTest,
  modelsStrList,
  modelHyperparams,
  sampleTechniquesStrList,
  normalizationTechniquesStrList,
):
  try:
    # Get the model.
    modelStr = trial.suggest_categorical("model", modelsStrList)

    # Create the hyperparameters dictionary.
    # Avoid: CategoricalDistribution does not support dynamic value space.
    hyperparams = {}
    for key, value in modelHyperparams[modelStr].items():
      hyperparams[key] = np.random.choice(value)

    trial.set_user_attr("hyperparams", str(hyperparams))

    # Get the oversampling technique.
    sampleTechniqueStr = trial.suggest_categorical("sampleTechnique", sampleTechniquesStrList)

    # Get the normalization technique.
    normTechniqueStr = trial.suggest_categorical("normTechnique", normalizationTechniquesStrList)

    # Sample the dataset.
    xTrain, yTrain, sampleObj = SampleDataset(xTrain, yTrain, techniqueStr=sampleTechniqueStr)

    # Normalize the data.
    xTrain, xTest, normObj = NormalizeData(xTrain, xTest, techniqueStr=normTechniqueStr)

    # Perform feature selection.
    xTrain, xTest, binIndices = FeaturesSelection(xTrain, xTest)
    trial.set_user_attr("binIndices", str(binIndices))

    # Train and evaluate the model.
    metrics, cm = TrainEvaluateModel(modelStr, hyperparams, xTrain, yTrain, xTest, yTest)

    # Calculate the objective value.
    objectiveValue = float(metrics["Weighted Accuracy"]) if ("Weighted Accuracy" in metrics) else metrics["Accuracy"]

  except Exception as e:
    objectiveValue = 0.0
    # print("Exception:", e)

  return objectiveValue


def TrainOptunaModel(
  xTrain,
  yTrain,
  xTest,
  yTest,
  modelsStrList,
  modelHyperparams,
  sampleTechniquesStrList,
  normalizationTechniquesStrList,
  nTrials=100,
  prefix="Optuna",
):
  # Create the study object.
  study = optuna.create_study(
    direction="maximize",  # To maximize the objective function.
    study_name=f"{prefix}_Study",  # The study name.
    storage=f"sqlite:///{prefix}_Study.db",  # The database file.
    load_if_exists=True,  # To load the study if it exists.
    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
    sampler=optuna.samplers.TPESampler(seed=42),  # Setting the sampler.
    # sampler=optuna.samplers.RandomSampler(seed=42),  # Setting the sampler.
  )

  # Create the objective function with the arguments.
  objectiveFunction = lambda trial: OptunaObjectiveFunction(
    trial,
    xTrain,
    yTrain,
    xTest,
    yTest,
    modelsStrList,
    modelHyperparams,
    sampleTechniquesStrList,
    normalizationTechniquesStrList,
  )
  # Optimize the objective function.
  study.optimize(
    objectiveFunction,  # Objective function.
    n_trials=nTrials,  # Number of trials.
    n_jobs=-1,  # To use all available CPUs for parallel execution.
    show_progress_bar=True,  # To show the progress bar.
  )

  # Store the trials in a dataframe.
  trials = study.trials_dataframe()
  trials.to_csv(f"{prefix}_Optuna_Trials_History.csv", index=False)

  # Get the best hyperparameters and the best value.
  bestParams = study.best_params
  bestValue = study.best_value
  print("Best Parameters:", bestParams)
  print("Best Value:", bestValue)

  # Save the best hyperparameters to a CSV file.
  bestParams = pd.DataFrame(bestParams, index=[0])
  bestParams.to_csv(f"{prefix}_Optuna_Best_Params.csv", index=False)

  # Save the study object.
  with open(f"{prefix}_Optuna_Study.pkl", "wb") as file:
    pickle.dump(study, file)

  return study, trials, bestParams, bestValue


modelsStrList = [
  "RandomForestClassifier",
  "DecisionTreeClassifier",
  "SVC",
  "KNeighborsClassifier",
  "MLPClassifier",
  "AdaBoostClassifier",
  "GradientBoostingClassifier",
  "ExtraTreesClassifier",
  "BaggingClassifier",
  "XGBClassifier",
]

modelHyperparams = {
  "RandomForestClassifier"    : {
    "n_estimators"     : [10, 50, 100, 200, 500, 1000, 2000],
    "max_features"     : ["sqrt", "log2", None],
    "max_depth"        : [None, 10, 50, 100, 200, 500, 1000],
    "min_samples_split": [2, 5, 10, 50, 100, 200],
    "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
    "bootstrap"        : [True, False],
  },
  "DecisionTreeClassifier"    : {
    "max_depth"        : [None, 10, 50, 100, 200, 500, 1000],
    "min_samples_split": [2, 5, 10, 50, 100, 200],
    "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
    "max_features"     : ["sqrt", "log2", None],
    "splitter"         : ["best", "random"],
  },
  "SVC"                       : {
    "C"                      : [0.01, 0.1, 1, 10, 100],
    "kernel"                 : ["linear", "poly", "rbf", "sigmoid"],
    "degree"                 : [2, 3, 4, 5, 6],
    "gamma"                  : ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "coef0"                  : [0.0, 0.1, 0.5, 1.0, 2.0],
    "shrinking"              : [True, False],
    "probability"            : [True, False],
    "class_weight"           : [None, "balanced"],
    "decision_function_shape": ["ovo", "ovr"],
  },
  "KNeighborsClassifier"      : {
    "n_neighbors": [3, 5, 7, 10, 15, 20],
    "weights"    : ["uniform", "distance"],
    "algorithm"  : ["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size"  : [10, 20, 30, 40, 50, 60],
    "p"          : [1, 2],
    "metric"     : ["minkowski", "euclidean", "manhattan"],
  },
  "MLPClassifier"             : {
    "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (100, 100, 50), (100, 100, 100)],
    "activation"        : ["identity", "logistic", "tanh", "relu"],
    "solver"            : ["lbfgs", "sgd", "adam"],
    "alpha"             : [0.0001, 0.001, 0.01, 0.1],
    "learning_rate"     : ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [0.001, 0.01, 0.1],
    "max_iter"          : [200, 300, 400, 500, 600],
    "early_stopping"    : [True, False],
  },
  "AdaBoostClassifier"        : {
    "n_estimators" : [10, 50, 100, 200, 500, 1000],
    "learning_rate": [0.01, 0.1, 1, 10],
    "algorithm"    : ["SAMME", "SAMME.R"],
  },
  "GradientBoostingClassifier": {
    "n_estimators"     : [10, 50, 100, 200, 500, 1000],
    "learning_rate"    : [0.01, 0.1, 0.5, 1, 10],
    "max_depth"        : [3, 5, 7, 9],
    "min_samples_split": [2, 5, 10, 50, 100],
    "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
    "subsample"        : [0.5, 0.7, 1.0],
    "max_features"     : ["auto", "sqrt", "log2", None],
  },
  "ExtraTreesClassifier"      : {
    "n_estimators"     : [10, 50, 100, 200, 500, 1000, 2000],
    "max_features"     : ["auto", "sqrt", "log2", None],
    "max_depth"        : [None, 10, 50, 100, 200, 500, 1000],
    "min_samples_split": [2, 5, 10, 50, 100, 200],
    "min_samples_leaf" : [1, 2, 5, 10, 50, 100],
    "bootstrap"        : [True, False],
  },
  "BaggingClassifier"         : {
    "n_estimators"      : [10, 50, 100, 200, 500, 1000],
    "max_samples"       : [0.1, 0.5, 1.0],
    "max_features"      : [0.1, 0.5, 1.0],
    "bootstrap"         : [True, False],
    "bootstrap_features": [True, False],
  },
  "XGBClassifier"             : {
    "n_estimators"    : [10, 50, 100, 200, 500, 1000],
    "learning_rate"   : [0.01, 0.1, 0.5, 1, 10],
    "max_depth"       : [3, 5, 7, 9],
    "min_child_weight": [1, 3, 5, 10],
    "subsample"       : [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5, 0.7, 1.0],
    "gamma"           : [0, 0.1, 0.5, 1],
    "reg_alpha"       : [0, 0.1, 1],
    "reg_lambda"      : [0, 0.1, 1],
  },
}

sampleTechniquesStrList = [
  None,
  "SMOTE",
  "ADASYN",
  "RandomOverSampler",
  "RandomUnderSampler",
  "NearMiss",
  "TomekLinks",
  "SMOTETomek",
  "SMOTEENN",
  "SVMSMOTE",
  "KMeansSMOTE",
  "BorderlineSMOTE",
]

normalizationTechniquesStrList = [
  None,
  "StandardScaler",
  "MinMaxScaler",
  "MaxAbsScaler",
  "RobustScaler",
  "Normalize_L1_0",
  "Normalize_L2_0",
  "Normalize_Max_0",
  "Normalize_L1_1",
  "Normalize_L2_1",
  "Normalize_Max_1",
]

fullPath = r"MedMNISTv2/MedMNISTv2_Nodule_MNIST_3D.csv"
trainPath = r"MedMNISTv2/MedMNISTv2_Nodule_MNIST_3D_Train.csv"
testPath = r"MedMNISTv2/MedMNISTv2_Nodule_MNIST_3D_Test.csv"
prefix = "MedMNISTv2_Nodule_MNIST_3D"

if (not os.path.exists(trainPath)):
  X, y, _, _ = LoadDataset(fullPath, fillValue=0)

  # Split the dataset into training and testing sets.
  xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

  # Store the dataset.
  dfTrain = pd.concat([xTrain, yTrain], axis=1)
  dfTrain.to_csv(trainPath, index=False)
  dfTest = pd.concat([xTest, yTest], axis=1)
  dfTest.to_csv(testPath, index=False)

# Load the dataset: xTrain, yTrain, xTest, yTest.
xTrain, yTrain, featuresColumns, targetColumn = LoadDataset(trainPath, fillValue=0)

xTest, yTest, _, _ = LoadDataset(testPath, fillValue=0)

# Encode the target data.
yTrain, encodeObj = EncodeData(yTrain, technique=LabelEncoder, transformOnly=False)
yTest, _ = EncodeData(yTest, technique=encodeObj, transformOnly=True)

# Train the Optuna model.
study, trials, bestParams, bestValue = TrainOptunaModel(
  xTrain,
  yTrain,
  xTest,
  yTest,
  modelsStrList,
  modelHyperparams,
  sampleTechniquesStrList,
  normalizationTechniquesStrList,
  nTrials=3000,
  prefix=prefix,
)

import numpy as np
import Orange
import Orange.regression
import Orange.classification
from Orange.classification import LogisticRegressionLearner
from Orange.classification import Model
from Orange.evaluation import CrossValidation
from Orange.data import Table, Domain, ContinuousVariable
from Orange.preprocess.preprocess import Preprocess

class RemoveConstant(Preprocess):
    def __call__(self,data):
        oks = np.min(data.X, axis=0) != np.max(data.X, axis=0)
        atts = [a for a, ok in zip(data.domain.attributes, oks) if ok]
        domain = Orange.data.Domain(atts, data.domain.class_vars)
        return Orange.data.Table(domain, data)

class StackedClassificationLearner(Orange.classification.Learner):
    def __init__(self, learners, meta_learner=LogisticRegressionLearner, k=5):
        super().__init__()
        self.k = k  # number of internal cross-validations
        self.learners = learners
        self.meta_learner = meta_learner  # base learner
        self.name = "stacking"

    def fit_storage(self, data):
        res = Orange.evaluation.CrossValidation(data, self.learners, self.k)
        X = np.column_stack(res.probabilities)
        n_classes = np.unique(data.Y).shape[0]
        metaDomain = Domain([ContinuousVariable(i) for i in range(n_classes*len(self.learners))],data.domain.class_var)
        table = Table(metaDomain,X,res.actual)
        resMeta = LogisticRegressionLearner(table)
        models = [learner(data) for learner in self.learners]

        return StackedClassificationModel(data.domain,metaDomain,models,resMeta)


class StackedClassificationModel(Orange.classification.Model):
    def __init__(self, domain, meta_domain, models,
                 meta_model, name="stacking"):
        super().__init__(domain)
        self.models = models  # a list of predictors
        self.meta_model = meta_model  # meta model
        self.meta_domain = meta_domain  # meta domain
        self.name = name

    def predict_storage(self, data):
        models = [model(data,1) for model in self.models]
        X = np.column_stack(models)
        table1 = Table(self.meta_domain,X,np.zeros(len(data.Y)))
        return self.meta_model(table1, Model.ValueProbs)

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    data = Orange.data.Table("digits-79.csv")
    #data = Orange.data.Table("digits-358.csv")
    knn = Orange.classification.KNNLearner()
    rf = Orange.classification.RandomForestLearner(n_estimators=20)
    lr = Orange.classification.LogisticRegressionLearner()
    sm = Orange.classification.SoftmaxRegressionLearner()
    svm = Orange.classification.LinearSVMLearner()
    learners = [knn, lr, rf, sm, svm]

    scl = StackedClassificationLearner(learners, k=5)
    res = Orange.evaluation.CrossValidation(data, learners + [scl], k=5)
    print(Orange.evaluation.AUC(res))

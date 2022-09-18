"""Code for the LSUN experiment.
"""
import random
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


class LSUN:
    """LSUN experiment.
    """

    def __init__(
        self,
        features,
        gt_labels,
    ):
        self.features = features
        self.gt_labels = gt_labels

        # classifier
        self.svm = None
        self.clf = None

        # obtained labels
        # 1  = positive
        # 0  = negative
        # -1 = unknown
        self.labels = None

        # low and high threshold
        self.thresholds = [None, None]

        # number of questions asked
        self.num_questions = 0

        # initialize
        self.init_classifier()
        self.init_labels()

    def init_classifier(self, dual=True):
        # TODO(ethan): decide dual True/False
        self.svm = LinearSVC(dual=dual, random_state=0, tol=1e-4, C=1.0)
        self.clf = CalibratedClassifierCV(self.svm)

    def init_labels(self):
        self.labels = -1.0 * np.ones(len(self.features)).astype("int")

    def sample_unknown(self, num_samples=100):
        """Returns a sample of images.
        If num_samples is None, then return all.
        """
        indices = list(np.argwhere(self.labels == -1)[:, 0])
        if num_samples is not None:
            indices = random.sample(indices, k=min(len(indices), num_samples))
        return indices

    def get_train_data_from_indices(self, indices):
        X = self.features[indices]
        y = self.gt_labels[indices]
        return X, y

    def set_gt(self, samples):
        """Set the Gt for these indices.
        """
        for idx in samples:
            assert self.labels[idx] == -1  # meaning unknown
            self.num_questions += 1
            self.labels[idx] = self.gt_labels[idx]

    def get_gt(self):
        """Returns the indices that have been labeled.
        """
        return list(np.argwhere(self.labels != -1)[:, 0])

    def get_pos_indices(self):
        """Returns indices with positive labels (correct).
        """
        return list(np.argwhere(self.labels == 1)[:, 0])

    def acc_of_labeled(self):
        """Returns percent of correct labels.
        """
        indices = self.get_gt()
        our_labels = self.labels[indices]
        gt_labels = self.gt_labels[indices]
        acc = (our_labels == gt_labels).sum() / len(our_labels)
        return acc

    def get_training_data(self):
        """Returns X, y for training.
        """
        indices = self.get_gt()
        return self.get_train_data_from_indices(indices)

    def get_inference_data(self):
        """Returns (X, y, indices) for inference (the unknown data points).
        """
        indices = self.sample_unknown(num_samples=None)
        X = self.features[indices]
        y = self.gt_labels[indices]
        return X, y, indices

    def update_pool(self, y_prob, indices):
        """Updates the pool given probs and indices.
        """
        pos_labels = np.argwhere(y_prob > self.thresholds[1])[:, 0]  # high confidence
        neg_labels = np.argwhere(y_prob < self.thresholds[0])[:, 0]  # low confidence
        for idx in pos_labels:
            assert self.labels[indices[idx]] == -1
            self.labels[indices[idx]] = 1
        for idx in neg_labels:
            assert self.labels[indices[idx]] == -1
            self.labels[indices[idx]] = 0
        num_added = len(pos_labels) + len(neg_labels)
        return num_added

    def sef_thresholds(self, y_prob, gt):
        indices = np.argsort(y_prob)
        probs = y_prob[indices]
        labels = gt[indices]
        positive = len(np.argwhere(labels == 1)[:, 0])
        negative = len(np.argwhere(labels == 0)[:, 0])

        def get_low_index(labels, threshold=0.01):
            percent_pos = None
            num = 0
            for idx in range(len(labels)):
                if labels[idx] == 1.0:
                    num += 1
                percent_pos = num / positive
                if percent_pos > threshold:
                    return probs[idx]
            print("Low thresh not found.")
            return 0.0

        def get_high_index(labels, threshold=0.01):
            for idx in range(len(labels)):
                num_gt_positives = np.array(labels[idx:]).sum()
                percent_gt_pos = num_gt_positives / len(labels[idx:])
                if percent_gt_pos >= threshold:
                    return probs[idx]
            print("High thresh not found.")
            return 1.0

        self.thresholds[0] = get_low_index(labels, threshold=0.01)
        self.thresholds[1] = get_high_index(labels, threshold=0.95)

    def run_loop(self, num_samples=200, percent_training=0.8):
        """
        (percent_training) for training
        (1 - percent_training) for choosing threshold

        Resources:
            - https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
        """
        print("\nRunning loop:")
        # sample from unknown pool
        samples = self.sample_unknown(num_samples=num_samples)
        if len(samples) > 0:
            print("Sampling {} annotations to label.".format(len(samples)))
        else:
            print("No more annotations to label! Can't sample anything.")
            return None
        # set new labels
        # print("Obtaining labels and updating pool.")

        num_train = int(len(samples) * percent_training)
        training_samples = samples[:num_train]
        validation_samples = samples[num_train:]

        self.set_gt(training_samples)
        dual = self.features.shape[1] > len(self.get_gt())
        self.init_classifier(dual=dual)

        # train the classifier
        print("Training the classifer.")
        X_train, y_train = self.get_training_data()
        self.clf.fit(X_train, y_train)

        # set the confidence score with validation set
        print("Updating the thresholds.")
        X_val, y_val = self.get_train_data_from_indices(validation_samples)
        y_prob = self.clf.predict_proba(X_val)[:, 1]  # prob of being
        self.sef_thresholds(y_prob, y_val)
        print("Thresholds: {}".format(self.thresholds))
        self.set_gt(validation_samples)

        # run inference on the others
        # print("Running inference on the unknown data points.")
        if len(self.sample_unknown(num_samples=None)) == 0:
            print("Finished!")
            return None
        X_test, y_test, indices = self.get_inference_data()
        y_prob = self.clf.predict_proba(X_test)[:, 1]  # prob of being
        num_added = self.update_pool(y_prob, indices)
        print(f"Added {num_added} labeled items to the pool.")
        return num_added

    def run(self,
            iters=5,
            sample_size=100,
            ious=None):
        data = {}
        data["num_labeled"] = []
        data["num_questions"] = []
        data["miou"] = []
        data["acc"] = []
        for i in tqdm(range(iters)):
            num_added = self.run_loop(num_samples=sample_size)
            if num_added is None:
                break
            num_labeled = len(self.get_pos_indices())
            data["num_labeled"].append(num_labeled)
            data["acc"].append(self.acc_of_labeled())
            data["num_questions"].append(self.num_questions)
            if ious is not None:
                pos_indices = self.get_pos_indices()
                miou = ious[pos_indices].mean()
                data["miou"].append(miou)
        return data

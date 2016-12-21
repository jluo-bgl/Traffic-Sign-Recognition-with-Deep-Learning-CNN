import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('fivethirtyeight')


class SignNames(object):
    def __init__(self, file_name):
        """
        DataFrame will be ClassId(Index), SignName
        :param file_name:
        """
        self.file_name = file_name
        self.data_frame = pd.read_csv(file_name, delimiter=',', encoding="utf-8-sig")
        self.data_frame = self.data_frame.set_index(["ClassId"])

    def sign_name_by_id(self, class_id):
        """
        Get Sign Name by class id
        :param class_id: int
        :return: String, the sign name
        """
        return self.data_frame.loc[class_id].values[0]


class DataExplorer(object):
    def __init__(self, sign_names, Train_Features, train_labels, Validation_Features, validation_labels, Test_Features, test_labels):
        """
        Constructor DataExplorer
        :param sign_names: ClassId and SignName
        :param Train_Features: training pictures
        :param train_labels: training labels
        :param Test_Features: testing pictures
        :param test_labels: testing labels
        """
        self.sign_names = sign_names
        self.Train_Features = Train_Features
        self.train_labels = train_labels
        self.Test_Features = Test_Features
        self.test_labels = test_labels
        self.Validation_Features = Validation_Features
        self.validation_labels = validation_labels

    @staticmethod
    def _sample(feature, labels, data_slice, sign_names):
        images = feature[data_slice]
        labels = labels[data_slice]
        image_count = images.shape[0]
        fig, axes = plt.subplots(image_count, 1, figsize=(3, 2 * image_count))
        pbar = tqdm(range(image_count), desc="showing image ", total=image_count)
        for index in pbar:
            image = images[index]
            class_id = labels[index]
            axes[index].set_title("(%d)%s" % (class_id, sign_names.sign_name_by_id(class_id)))
            axes[index].imshow(image)

        fig.tight_layout()
        return plt

    def sample_training_data(self, data_slice):
        return self._sample(self.Train_Features, self.train_labels, data_slice, self.sign_names)

    def sample_testing_data(self, data_slice):
        return self._sample(self.Test_Features, self.test_labels, data_slice, self.sign_names)

    @staticmethod
    def _data_distribution(labels, sign_names):
        labels_df = pd.DataFrame({"ClassId": labels})
        merged_df = pd.merge(labels_df, sign_names.data_frame, left_on="ClassId", right_index=True, how="left")
        distribution = merged_df.groupby("SignName").count()
        return distribution

    def training_data_distribution(self):
        return self._data_distribution(self.train_labels, self.sign_names)

    def validation_data_distribution(self):
        return self._data_distribution(self.validation_labels, self.sign_names)

    def testing_data_distribution(self):
        return self._data_distribution(self.test_labels, self.sign_names)

    def bar_chart_data_distribution(self, distribution, title):
        """
        show a bar chart for data distribution
        :param distribution:DataFrame distribution returned from training_data_distribution or testing_data_distribution
        :param title: the title you'd like to show
        :return: the polt
        """
        plt.figure(figsize=(10, 7))
        y_pos = np.arange(len(distribution))
        plt.barh(y_pos, distribution['ClassId'])
        plt.yticks(y_pos, distribution.index)
        plt.xlabel('Count'),
        ax = plt.gca()
        ax.tick_params(axis='x', which='both', labelsize=8)
        ax.tick_params(axis='y', which='both', labelsize=8)
        plt.title(title)
        plt.gcf().tight_layout()
        return plt

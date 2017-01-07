import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from .laplotter import LossAccPlotter
from datetime import datetime
from PIL import Image
import math

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

    def names(self):
        return self.data_frame.values.reshape(-1)


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
    def from_data_provider(sign_names, provider):
        return DataExplorer(
            sign_names = sign_names,
            Train_Features = provider.X_train,
            train_labels = provider.y_train,
            Validation_Features = provider.X_validation,
            validation_labels = provider.y_validation,
            Test_Features = provider.X_test,
            test_labels = provider.y_test
        )

    @staticmethod
    def _sample(feature, labels, data_slice, sign_names):
        images = feature[data_slice]
        labels = labels[data_slice]
        image_count = len(images)
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

    def sample_validation_data(self, data_slice):
        return self._sample(self.Validation_Features, self.validation_labels, data_slice, self.sign_names)

    def sample_testing_data(self, data_slice):
        return self._sample(self.Test_Features, self.test_labels, data_slice, self.sign_names)

    @staticmethod
    def _data_distribution(labels, sign_names):
        labels_df = pd.DataFrame({"ClassId": labels})
        merged_df = pd.merge(labels_df, sign_names.data_frame, left_on="ClassId", right_index=True, how="left")
        distribution = merged_df.groupby("SignName").count()
        return distribution

    def training_data_distribution(self):
        """
        training data distribution
        :return: the distribution, groupby("signName").count()
        """
        return self._data_distribution(self.train_labels, self.sign_names)

    def validation_data_distribution(self):
        return self._data_distribution(self.validation_labels, self.sign_names)

    def testing_data_distribution(self):
        return self._data_distribution(self.test_labels, self.sign_names)

    @staticmethod
    def highest_sign_names_count(distribution):
        """

        :param distribution: the distribution to exam
        :return: sign_name, count
        """
        idx = distribution.idxmax(axis=0)
        return distribution.loc[idx].iloc[0].name, distribution.loc[idx].iloc[0][0]

    @staticmethod
    def lowest_sign_names_count(distribution):
        """

        :param distribution: the distribution to exam
        :return: sign_name, count
        """
        idx = distribution.idxmin(axis=0)
        return distribution.loc[idx].iloc[0].name, distribution.loc[idx].iloc[0][0]

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

    @staticmethod
    def _summary_array(data):
        return """
            examples:{}
            shape:{}
        """.format(len(data), data.shape)

    def _all_labels(self):
        labels = np.append(np.unique(self.train_labels), np.unique(self.validation_labels))
        labels = np.unique(np.append(labels, np.unique(self.test_labels)))
        return labels

    def summary(self):
        return """
        training data set: {}
        validation data set: {}
        testing data set: {}
        unique classes: {}
        """.format(DataExplorer._summary_array(self.Train_Features),
                   DataExplorer._summary_array(self.Validation_Features),
                   DataExplorer._summary_array(self.Test_Features),
                   len(self._all_labels()))


class TrainingPlotter(object):
    def __init__(self, title, file_name, show_plot_window=False):
        self.plotter = LossAccPlotter(title=title,
                                      save_to_filepath=file_name,
                                      show_regressions=True,
                                      show_averages=True,
                                      show_loss_plot=True,
                                      show_acc_plot=True,
                                      show_plot_window=show_plot_window,
                                      x_label="Epoch")

    def add_loss_accuracy_to_plot(self, epoch, loss_train, acc_train, loss_val, acc_val, redraw=True):
        self.plotter.add_values(epoch, loss_train=loss_train, acc_train=acc_train, loss_val=loss_val, acc_val=acc_val,
                                redraw=redraw)
        return self.plotter.fig

    def safe_shut_down(self):
        self.plotter.block()

    @staticmethod
    def now_as_str():
        return "{:%Y_%m_%d_%H_%M}".format(datetime.now())

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        from sklearn.metrics import confusion_matrix
        cmap = plt.cm.binary
        cm = confusion_matrix(y_true, y_pred)
        tick_marks = np.array(range(len(labels))) + 0.5
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(20, 16), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        intFlag = 0
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            if (intFlag):
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')
            else:
                c = cm_normalized[y_val][x_val]
                if (c > 0.01):
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
        if (intFlag):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('')
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')
        return plt

    @staticmethod
    def combine_images(images, file_name, top_images=1500):
        if len(images) > top_images:
            images = images[0:top_images-1]
        count = len(images)
        max_images_pre_row = 50
        width = max_images_pre_row * 32
        heigh = math.ceil(count / max_images_pre_row) * 32

        blank_image = Image.new("RGB", (width, heigh))
        for index in range(count):
            image = Image.fromarray(images[index])
            column = index % max_images_pre_row
            row = index // max_images_pre_row
            blank_image.paste(image, (column * 32, row * 32))
        blank_image.save(file_name)

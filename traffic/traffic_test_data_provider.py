from .traffic_data import TrafficDataProviderAutoSplitValidationData
from .traffic_data import TrafficDataRealFileProviderAutoSplitValidationData
from .traffic_data_enhance import *

real_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(split_validation_from_train=True)
real_data_provider_no_shuffer = TrafficDataRealFileProviderAutoSplitValidationData(split_validation_from_train=False)
clear_data_range = slice(80, 90)
clear_subset_data_provider = TrafficDataProviderAutoSplitValidationData(
    real_data_provider_no_shuffer.X_train[clear_data_range],
    real_data_provider_no_shuffer.y_train[clear_data_range],
    real_data_provider_no_shuffer.X_test[clear_data_range],
    real_data_provider_no_shuffer.y_test[clear_data_range],
    split_validation_from_train=False
)


def real_data_provider_enhanced_with_random_rotate(ratio):
    _X_train, _y_train = enhance_with_random_rotate(
        real_data_provider_no_shuffer.X_train,
        real_data_provider_no_shuffer.y_train,
        ratio)
    real_data_provider_enhanced_value = TrafficDataProviderAutoSplitValidationData(
        _X_train,
        _y_train,
        real_data_provider_no_shuffer.X_test,
        real_data_provider_no_shuffer.y_test,
        split_validation_from_train=True
    )
    return real_data_provider_enhanced_value


def real_data_provider_enhanced_with_random_zoomin(ratio):
    _X_train, _y_train = enhance_with_random_zoomin(real_data_provider_no_shuffer.X_train,
                                                                           real_data_provider_no_shuffer.y_train, ratio)
    real_data_provider_enhanced_value = TrafficDataProviderAutoSplitValidationData(
        _X_train,
        _y_train,
        real_data_provider_no_shuffer.X_test,
        real_data_provider_no_shuffer.y_test,
        split_validation_from_train=True
    )
    return real_data_provider_enhanced_value


def real_data_provider_enhanced_with_random_rotate_and_zoomin(ratio):
    _X_train, _y_train = enhance_with_random_zoomin_and_rotate(
        real_data_provider_no_shuffer.X_train,
        real_data_provider_no_shuffer.y_train,
        ratio)
    real_data_provider_enhanced_value = TrafficDataProviderAutoSplitValidationData(
        _X_train,
        _y_train,
        real_data_provider_no_shuffer.X_test,
        real_data_provider_no_shuffer.y_test,
        split_validation_from_train=True
    )
    return real_data_provider_enhanced_value
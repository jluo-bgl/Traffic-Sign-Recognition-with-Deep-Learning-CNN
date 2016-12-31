from .traffic_data import TrafficDataProvider
from .traffic_data import TrafficDataRealFileProvider
from .traffic_data import TrafficDataEnhancement

real_data_provider = TrafficDataRealFileProvider(split_validation_from_train=True)
real_data_provider_no_shuffer = TrafficDataRealFileProvider(split_validation_from_train=False)
clear_data_range = slice(80, 90)
clear_subset_data_provider = TrafficDataProvider(
    real_data_provider_no_shuffer.X_train[clear_data_range],
    real_data_provider_no_shuffer.y_train[clear_data_range],
    real_data_provider_no_shuffer.X_test[clear_data_range],
    real_data_provider_no_shuffer.y_test[clear_data_range],
    split_validation_from_train=False
)

_X_train, _y_train = TrafficDataEnhancement.enhance(real_data_provider_no_shuffer.X_train,
                                                    real_data_provider_no_shuffer.y_train)
real_data_provider_enhanced = TrafficDataProvider(
    _X_train,
    _y_train,
    real_data_provider_no_shuffer.X_test,
    real_data_provider_no_shuffer.y_test,
    split_validation_from_train=True
)
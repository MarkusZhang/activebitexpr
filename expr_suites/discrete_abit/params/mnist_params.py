import platform
_IS_LINUX = platform.system() == "Linux"

########################################
# parameters for image preprocessing
########################################
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

########################################
# parameters for the model
########################################
hash_size = 16

########################################
# settings for training
########################################
# image-scale: 100 for office, 28 for mnist
image_scale = 28
shuffle_batch = True

## source data path
# mini-office: "F:/data/domain_adaptation_images/train-test-split-for-domadv/mini/train/source"
_windows_src_path = "F:/data/mnist/mini/training"
_linux_src_path = "/home/jdchen/image_hash/data/mnist/mini/training"
source_data_path = _linux_src_path if _IS_LINUX else _windows_src_path

## target data path
# mini-office: "F:/data/domain_adaptation_images/train-test-split-for-domadv/mini/train/target"
_windows_tgt_path = "F:/data/mnist_m/mini/train-10-percent"
_linux_tgt_path = "/home/jdchen/image_hash/data/mnist_m/super-mini/train"
target_data_path = _linux_tgt_path if _IS_LINUX else _windows_tgt_path

iterations = 100
batch_size = 50
learning_rate = 0.001 #TODO: tunable

loss_coeffs = {
    "abit_res": 0.25,
    "abit_pair": 0.25,
    "h_mmd": 0.25,
    "h_pair": 0.25
}
########################################
# settings for ml_test
########################################

# test data path
_windows_test_path = {
    "query": "F:/data/mnist_m/query-db-split/query",
    "db": "F:/data/mnist_m/query-db-split/db"
}
_linux_test_path = {
    "query": "/home/jdchen/image_hash/data/mnist_m/mnist_m/test/query",
    "db": "/home/jdchen/image_hash/data/mnist_m/mnist_m/test/db"
}
test_data_path = _linux_test_path if _IS_LINUX else _windows_test_path

test_batch_size = 100
# in ml_test, retrieval precision within hamming radius `precision_radius` will be calculated
precision_radius = 2
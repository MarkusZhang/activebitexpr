import itertools
import os, sys

import torch
from ml_toolkit.pytorch_utils import loss
from ml_toolkit.pytorch_utils.misc import autocuda, get_data_loader, hash_func_wrapper
from ml_toolkit.pytorch_utils.test_utils import run_test, load_models
from torch.autograd import Variable

from commons.ml_test.testing import run_simple_test
from commons.ml_test.testing_utils import save_test_results
from commons.ml_train import utils


sys.path.append(os.path.dirname(__file__))

def get_loss(basic_ext, code_gen, abit_gen, images, labels, loss_coeffs):
    """
    :param basic_ext, code_gen, abit_gen: nn module instances
    :param images: [source_images, tgt_images]
    :param labels: [source_labels, tgt_labels]
    :param dlabels: domain labels
    :param loss_coeffs: dict
    :return: total loss
    """
    src_imgs, tgt_imgs = images
    src_labels, tgt_labels = labels
    basic_feats_src, basic_feats_tgt = basic_ext(src_imgs), basic_ext(tgt_imgs)
    z_src, z_tgt = code_gen(basic_feats_src), code_gen(basic_feats_tgt)
    a_src,a_tgt = abit_gen(basic_feats_src), abit_gen(basic_feats_tgt)
    K = len(a_src[0])  # length of active bit vector
    h_src, h_tgt = torch.mul(a_src, z_src), torch.mul(a_tgt, z_tgt)    # the output code

    # active bit restriction loss, restricting active bit vector to only contain 1 zero
    abit_res_loss = torch.mean(torch.pow(
        (loss.get_L1_norm(torch.cat([a_src, a_tgt])) - (K - 1)) / K, 2
    ))

    # active bit pairwise loss, to make active bits of the same class the same
    a_src_shifted = a_src * 2 - 1 # convert it into the range of (-1, 1)
    a_tgt_shifted = a_tgt * 2 - 1
    abit_pair_loss = loss.get_pairwise_sim_loss(feats=torch.cat([a_src_shifted,a_tgt_shifted]), labels=torch.cat([src_labels,tgt_labels]), num_classes=10)

    # MMD loss on h, to minimize domain difference
    #TODO: pairwise loss is used temporarily, later replace with MMD
    h_mmd_loss = loss.get_crossdom_pairwise_sim_loss(src_feats=h_src, tgt_feats=h_tgt,src_labels=src_labels,tgt_labels=tgt_labels,
                                                     num_classes=10)

    # pairwise loss on h, to make code of the same class similar
    h_pair_loss = loss.get_pairwise_sim_loss(feats=torch.cat([h_src,h_tgt]), labels=torch.cat([src_labels,tgt_labels]),num_classes=10)

    c = loss_coeffs
    total_loss = c["abit_res"] * abit_res_loss + c["abit_pair"] * abit_pair_loss \
                 + c["h_mmd"] * h_mmd_loss + c["h_pair"] * h_pair_loss

    return total_loss

def train(params,logger, basic_ext, code_gen, abit_gen):
    "return the trained model, and loss records"
    source_loader = utils.get_data_loader(data_path=params.source_data_path, params=params)
    target_loader = utils.get_data_loader(data_path=params.target_data_path, params=params)

    opt_basic_feat = torch.optim.Adam(basic_ext.parameters(), lr=params.learning_rate)
    opt_code_gen = torch.optim.Adam(code_gen.parameters(), lr=params.learning_rate)
    opt_abit_gen = torch.optim.Adam(abit_gen.parameters(), lr=params.learning_rate)

    total_loss_records = []

    for i in range(params.iterations):
        # refresh data loader
        itertools.tee(source_loader)
        itertools.tee(target_loader)
        zipped_loader = enumerate(zip(source_loader, target_loader))
        acc_total_loss = 0
        logger.info("epoch {}/{} started".format(i,params.iterations))
        # train using minibatches
        for step, ((images_src, labels_src), (images_tgt, labels_tgt)) in zipped_loader:
            logger.info("batch {}".format(step))
            min_len = min(len(images_src),len(images_tgt))
            images_src = autocuda(Variable(images_src[:min_len]).float())
            images_tgt = autocuda(Variable(images_tgt[:min_len]).float())
            labels_src = autocuda(labels_src[:min_len])
            labels_tgt = autocuda(labels_tgt[:min_len])

            # clear gradients
            opt_basic_feat.zero_grad(); opt_code_gen.zero_grad(); opt_abit_gen.zero_grad()

            # calc loss
            total_loss = get_loss(basic_ext=basic_ext,code_gen=code_gen,abit_gen=abit_gen,images=[images_src, images_tgt],
                     labels=[labels_src,labels_tgt],loss_coeffs=params.loss_coeffs)

            # make updates
            total_loss.backward()
            opt_basic_feat.step(); opt_code_gen.step(); opt_abit_gen.step()

            acc_total_loss += total_loss.cpu().data.numpy()[0]

        total_loss_records.append(acc_total_loss / (step + 1))

    return {
        "loss_records": total_loss_records,
        "models": {
            "basic_ext": basic_ext,
            "code_gen": code_gen,
            "abit_gen": abit_gen
        }
    }


def main(model_def, params, do_training=True):

    save_model_to = "saved_models"

    if (do_training):
        # instantiate models
        code_gen = autocuda(model_def.CodeGen(params))
        abit_gen = autocuda(model_def.AbitGen(params))
        basic_ext = autocuda(model_def.BasicExt(params))

        # run training
        logger = utils.LoggerGenerator.get_logger(log_file_path="log/log.txt")
        train_results = train(params=params,logger=logger,basic_ext=basic_ext,code_gen=code_gen,abit_gen=abit_gen)

        # save models and records
        utils.save_models(models=train_results["models"],save_model_to=save_model_to)
        utils.save_loss_records(loss_records=train_results["loss_records"],loss_name="total_loss",save_to=save_model_to)

    # run testing
    query_loader = get_data_loader(data_path=params.test_data_path["query"],dataset_mean=params.dataset_mean,dataset_std=params.dataset_std,
                                   batch_size=params.batch_size,shuffle_batch=False,image_scale=params.image_scale)
    db_loader = get_data_loader(data_path=params.test_data_path["db"], dataset_mean=params.dataset_mean,
                                   dataset_std=params.dataset_std, batch_size=params.batch_size, shuffle_batch=False,image_scale=params.image_scale)
    models = load_models(path=save_model_to,model_names=["basic_ext","code_gen","abit_gen"])
    hash_model = model_def.get_hash_function(basic_ext=models["basic_ext"],code_gen=models["code_gen"],
                                abit_gen = models["abit_gen"])
    hash_model = hash_func_wrapper(hash_model)
    test_results = run_test(query_loader=query_loader,db_loader=db_loader,hash_model=hash_model,radius=params.precision_radius)
    save_test_results(test_results=test_results,save_to=os.path.join(save_model_to,"test_results"))

########################################################################################
##############    The following are experiment suites to run  ##########################
########################################################################################

def suite_1():
    "MNIST training and testing"
    from models import mnist_model as model_def
    from params import mnist_params
    main(model_def=model_def,params=mnist_params,do_training=True)


def mnist_test_on_training_data():
    from models import mnist_model as model_def
    from params import mnist_params
    mnist_params.test_data_path = {
        "query": "/home/jdchen/image_hash/data/mnist_m/mnist_m/test/query",
        "db": "/home/jdchen/image_hash/data/mnist_m/mnist_m/test/db"
    }
    main(model_def=model_def, params=mnist_params, do_training=False)


if __name__ == "__main__":
    suite_1()
    # mnist_test_on_training_data()

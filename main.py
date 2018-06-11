import tensorflow as tf

from utils import mkdir_p
from PGGAN import PGGAN
from utils import CelebA
flags = tf.app.flags
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

flags.DEFINE_string("OPER_NAME", "Experiment_5_28_2", "the name of experiments")
flags.DEFINE_integer("OPER_FLAG", 0, "Flag of opertion: 0 is for training ")
flags.DEFINE_string("path" , '/home/?/data/celebA/', "Path of training data, for example /home/hehe/celebA/")
flags.DEFINE_integer("batch_size", 16, "Batch size")
flags.DEFINE_integer("sample_size", 512, "Size of sample")
flags.DEFINE_integer("max_iters", 40000, "Maxmization of training number")
flags.DEFINE_float("learn_rate", 0.001, "Learning rate for G and D networks")
flags.DEFINE_integer("lam_gp", 10, "Weight of gradient penalty term")
flags.DEFINE_integer("lam_eps", 0.001, "Weight for the epsilon term")
flags.DEFINE_float("flag", 11, "FLAG of gan training process")
flags.DEFINE_boolean("use_wscale", True, "Using the scale of weight")

FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = "./PGGanCeleba/logs/celeba_test2"
    mkdir_p(root_log_dir)

    OPER_NAME = FLAGS.OPER_NAME
    OPER_FLAG = FLAGS.OPER_FLAG

    data_In = CelebA(FLAGS.path)

    print ("the num of dataset", len(data_In.image_list))

    if OPER_FLAG == 0:

        fl = [1,2,2,3,3,4,4,5,5, 6, 6]
        r_fl = [1,1,2,2,3,3,4,4,5, 5, 6]

        for i in range(FLAGS.flag):

            t = False if (i % 2 == 0) else True
            pggan_checkpoint_dir_write = "./PGGanCeleba{}/model_pggan_{}/{}/".format(OPER_NAME, OPER_FLAG, fl[i])
            sample_path = "./PGGanCeleba{}/{}/sample_{}_{}".format(OPER_NAME, FLAGS.OPER_FLAG, fl[i], t)
            mkdir_p(pggan_checkpoint_dir_write)
            mkdir_p(sample_path)
            pggan_checkpoint_dir_read = "./PGGanCeleba{}/model_pggan_{}/{}/".format(OPER_NAME, OPER_FLAG, r_fl[i])

            pggan = PGGAN(batch_size=FLAGS.batch_size, max_iters=FLAGS.max_iters,
                            model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                            data=data_In, sample_size=FLAGS.sample_size,
                            sample_path=sample_path, log_dir=root_log_dir, learn_rate=FLAGS.learn_rate, lam_gp=FLAGS.lam_gp, lam_eps=FLAGS.lam_eps, PG= fl[i],
                            t=t, use_wscale=FLAGS.use_wscale)

            pggan.build_model_PGGan()
            pggan.train()












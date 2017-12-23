import tensorflow as tf

from utils import mkdir_p
from PGGAN import PGGAN
from utils import CelebA
flags = tf.app.flags

flags.DEFINE_integer("OPER_FLAG", 0, "the flag of opertion: 0 is for training ")
flags.DEFINE_string("path" , '?', "the path of training data, for example /home/hehe/celebA/")
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("max_iters", 32000, "the maxmization of training number")
flags.DEFINE_float("learn_rate", 0.0001, "the learning rate for G and D networks")
flags.DEFINE_float("flag", 9, "the FLAG of gan training process")

FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = "./PGGanCeleba/logs/celeba_test2"
    mkdir_p(root_log_dir)
    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters
    sample_size = 512
    GAN_learn_rate = FLAGS.learn_rate

    OPER_FLAG = FLAGS.OPER_FLAG
    data_In = CelebA(FLAGS.path)
    print "the num of dataset", len(data_In.image_list)

    if OPER_FLAG == 0:

        fl = [1,2,2,3,3,4,4,5,5, 6, 6]
        r_fl = [1,1,2,2,3,3,4,4,5, 5, 6]

        for i in range(FLAGS.flag):

            t = False if (i % 2 == 0) else True
            pggan_checkpoint_dir_write = "./model_pggan_{}/{}/model.ckpt".format(OPER_FLAG, fl[i])
            sample_path = "./PGGanCeleba/{}/sample_{}_{}".format(FLAGS.OPER_FLAG, fl[i], t)
            mkdir_p(pggan_checkpoint_dir_write)
            mkdir_p(sample_path)
            pggan_checkpoint_dir_read = "./model_pggan_{}/{}/model.ckpt".format(OPER_FLAG, r_fl[i])

            pggan = PGGAN(batch_size=batch_size, max_iters=max_iters,
                            model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                            data=data_In, sample_size=sample_size,
                            sample_path=sample_path, log_dir=root_log_dir, learn_rate=GAN_learn_rate, PG= fl[i],
                            t=t)

            pggan.build_model_PGGan()
            pggan.train()












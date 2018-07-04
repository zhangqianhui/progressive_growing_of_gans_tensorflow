import tensorflow as tf
from ops import lrelu, conv2d, fully_connect, upscale, Pixl_Norm, downscale2d, MinibatchstateConcat
from utils import save_images
import numpy as np
from scipy.ndimage.interpolation import zoom

class PGGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, read_model_path, data, sample_size, sample_path, log_dir,
                 learn_rate, lam_gp, lam_eps, PG, t, use_wscale, is_celeba):
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gan_model_path = model_path
        self.read_model_path = read_model_path
        self.data_In = data
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.lam_gp = lam_gp
        self.lam_eps = lam_eps
        self.pg = PG
        self.trans = t
        self.log_vars = []
        self.channel = self.data_In.channel
        self.output_size = 4 * pow(2, PG - 1)
        self.use_wscale = use_wscale
        self.is_celeba = is_celeba
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False,name='alpha_tra')

    def build_model_PGGan(self):
        self.fake_images = self.generate(self.z, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.D_pro_logits = self.discriminate(self.images, reuse=False, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True,pg= self.pg, t=self.trans, alpha_trans=self.alpha_tra)

        # the defination of loss for D and G
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # gradient penalty from WGAN-GP
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits= self.discriminate(interpolates, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # 2 norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        self.D_origin_loss = self.D_loss
        self.D_loss += self.lam_gp * self.gradient_penalty
        self.D_loss += self.lam_eps * tf.reduce_mean(tf.square(self.D_pro_logits))

        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print ("The total para of D", total_para)

        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        print ("The total para of G", total_para2)

        #save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.output_size) not in var.name]

        print ("d_vars", len(self.d_vars))
        print ("g_vars", len(self.g_vars))

        print ("self.d_vars_n_read", len(self.d_vars_n_read))
        print ("self.g_vars_n_read", len(self.g_vars_n_read))

        print ("d_vars_n_2_rgb", len(self.d_vars_n_2_rgb))
        print ("g_vars_n_2_rgb", len(self.g_vars_n_2_rgb))

        # for n in self.d_vars:
        #     print (n.name)

        self.g_d_w = [var for var in self.d_vars + self.g_vars if 'bias' not in var.name]

        print ("self.g_d_w", len(self.g_d_w))

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)

        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            if self.pg != 1 and self.pg != 7:
                if self.trans:
                    self.r_saver.restore(sess, self.read_model_path)
                    self.rgb_saver.restore(sess, self.read_model_path)

                else:
                    self.saver.restore(sess, self.read_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iters:
                # optimization D
                n_critic = 1
                if self.pg >= 5:
                    n_critic = 1

                for i in range(n_critic):
                    sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                    if self.is_celeba:
                        train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                        realbatch_array = self.data_In.getShapeForData(train_list, resize_w=self.output_size)
                    else:
                        realbatch_array = self.data_In.getNextBatch(self.batch_size, resize_w=self.output_size)
                        realbatch_array = np.transpose(realbatch_array, axes=[0, 3, 2, 1]).transpose([0, 2, 1, 3])

                    if self.trans and self.pg != 0:
                        alpha = np.float(step) / self.max_iters
                        low_realbatch_array = zoom(realbatch_array, zoom=[1, 0.5, 0.5, 1], mode='nearest')
                        low_realbatch_array = zoom(low_realbatch_array, zoom=[1, 2, 2, 1], mode='nearest')
                        realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                    sess.run(opti_D, feed_dict={self.images: realbatch_array, self.z: sample_z})
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.z: sample_z})

                summary_str = sess.run(summary_op, feed_dict={self.images: realbatch_array, self.z: sample_z})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})

                if step % 400 == 0:
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.images: realbatch_array, self.z: sample_z})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2],
                                '{}/{:02d}_real.jpg'.format(self.sample_path, step))

                    if self.trans and self.pg != 0:
                        low_realbatch_array = np.clip(low_realbatch_array, -1, 1)
                        save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2],
                                    '{}/{:02d}_real_lower.jpg'.format(self.sample_path, step))
                   
                    fake_image = sess.run(self.fake_images,
                                          feed_dict={self.images: realbatch_array, self.z: sample_z})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.jpg'.format(self.sample_path, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.gan_model_path)

                step += 1

            save_path = self.saver.save(sess, self.gan_model_path)
            print ("Model saved in file: %s" % save_path)

        tf.reset_default_graph()

    def discriminate(self, conv, reuse=False, pg=1, t=False, alpha_trans=0.01):
        #dis_as_v = []
        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,
                           name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [self.batch_size, -1])

            #for D
            output = fully_connect(conv, output_size=1, use_wscale=self.use_wscale, gain=1, name='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def generate(self, z_var, pg=1, t=False, alpha_trans=0.0):
        with tf.variable_scope('generator') as scope:

            de = tf.reshape(Pixl_Norm(z_var), [self.batch_size, 1, 1, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=np.sqrt(2)/4, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [self.batch_size, 4, 4, int(self.get_nf(1))])
            de = conv2d(de, output_dim=self.get_nf(1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            for i in range(pg - 1):
                if i == pg - 2 and t:
                    #To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_1_{}'.format(de.shape[1]))))
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, use_wscale=self.use_wscale, name='gen_n_conv_2_{}'.format(de.shape[1]))))

            #To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, gain=1, name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            if pg == 1:
                return de

            if t:
                de = (1 - alpha_trans) * de_iden + alpha_trans*de

            else:
                de = de

            return de

    def get_nf(self, stage):
        return min(1024 / (2 **(stage * 1)), 512)

    def get_fp(self, pg):
        return max(512 / (2 **(pg - 1)), 16)

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps











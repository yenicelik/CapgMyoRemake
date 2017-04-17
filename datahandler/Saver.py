import tensorflow as tf
import os

class Saver(object):
    """ Handles saving and loading the trained weights
    """

    def __init__(self, save_dir):
        """
        :param parameter:
        :param W:
        :param b:
        :param save_dir:
        :return:
        """
        self.saver = tf.train.Saver()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #TODO:whenever loading the weights into it, we must unpack botch dictionaries and load them back in
        #TODO: create a folder, then save the parameters file to that folder


    def get_saver(self):
        return self.saver

    def save_session(self, sess, save_dir, global_step=None):
        """
        :param sess:
        :param save_dir:
        :param step:
        :param epoch:
        :param loss:
        :return:
        """
        checkpoint_path = os.path.join(save_dir, 'model.ckpt')
        if global_step is None:
            save_path = self.saver.save(sess, checkpoint_path)
        else:
            save_path = self.saver.save(sess, checkpoint_path, global_step=global_step)
        return True

    def load_session(self, sess, save_dir):
        """
         This must be called instead of 'sess.run(init)'
        :param sess:
        :param save_dir:
        :return:
        """
        checkpoint_path = os.path.join(save_dir, 'model.ckpt')
        print("Checkpoint path: ", checkpoint_path)
        checkpoint_path = os.path.join("./", checkpoint_path)
        print("Checkpoint path: ", checkpoint_path)
        self.saver.restore(sess, checkpoint_path)
        return True


if __name__ == '__main__':
    parameter = {
            'NUM_EPOCHS': 1,
            'BATCH_SIZE': 100
        }



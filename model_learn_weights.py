import tensorflow as tf
import numpy as np
from scipy import sparse


class LearnModel(object):
    def __int__(self, rule_length, training_iteration, learning_rate, fact_dic, entity_size):
        self.rule_length = rule_length
        self.training_iteration = training_iteration
        self.learning_rate = learning_rate
        self.model_name = 'learnModel_trainIter=%d_lr=%f.weights' % (training_iteration, learning_rate)
        self.fact_dic = fact_dic
        self.entity_size = entity_size

    def load_data(self, candidate, pt):
        self.rule_num = len(candidate)
        self.train_body = np.array(candidate)
        self.train_head = pt
        print("\nself.traindata:")
        print(self.train_body)

    def get_matrix(self, p):
        # sparse matrix
        pfacts = self.fact_dic.get(p)
        pmatrix = sparse.dok_matrix((self.entity_size, self.entity_size), dtype=np.int32)
        for f in pfacts:
            pmatrix[f[0], f[1]] = 1
        return pmatrix

    def train(self):
        # define the model parameters
        rule_index = tf.constant([i for i in range(self.rule_length)])
        x_body = tf.placeholder(shape=[self.rule_length], dtype=tf.int32)
        y_head = tf.placeholder(shape=[1], dtype=tf.int32)
        w = tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32))

        # x_body = self.train_body[a]
        # print("Rule body : " + str(x_body))
        # y_head = self.train_head
        s = tf.Session()
        body = rule_index.eval(session=s)
        print("out2=", type(body))

        # get matrix operation
        M_R = None
        index = -1
        for i in range(self.rule_length):
            if index == -1:
                M_R = self.get_matrix(body[i])
                index = 0
            else:
                M_R = sparse.dok_matrix(np.dot(M_R, self.get_matrix(body[i])))
        M_R_t = self.get_matrix(y_head)

        # define loss and train_step
        loss = 0
        for key in M_R.keys():
            loss = loss + np.square(1 / w * M_R[key[0], key[1]] - M_R_t[key[0], key[1]])
            M_R_t[key[0], key[1]] = -999
        for key in M_R_t.keys():
            if M_R_t[key[0], key[1]] != -999:
                loss = loss + np.square(1 / w * M_R[key[0], key[1]] - M_R_t[key[0], key[1]])
        loss = tf.sqrt(loss)
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        # AdagradOptimizer

        loss_history = []
        # try another batch training!!!!!!!!!
        with tf.Graph().as_default(), tf.Session() as sess:
            print("Training begins.")
            # initialize all variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for step in range(self.training_iteration):
                temp_loss = 0
                # every iteration is for all data
                print("Iteration %d: " % step)
                for i in range(self.rule_num):
                    sess.run(train_step, feed_dict={x_body: self.train_body[i], y_head: self.train_head})
                    print(' w = ' + str(sess.run(w)))
                    temp_loss = temp_loss + sess.run(loss, feed_dict={x_body: self.train_body[i],
                                                                      y_head: self.train_head})

                    # print("index: " + str(sess.run(rule_index)))
                # if (i + 1) % 5 == 0:
                print('loss = ' + str(temp_loss) + "\n")
                loss_history.append(temp_loss)
            saver = tf.train.Saver()
            saver.save(sess, './weights/%s.temp' % "pt=" + str(self.train_head) + "_modelxxxxx")  # modify!
            print("Training ends.")

    def getWeights(self):
        pass

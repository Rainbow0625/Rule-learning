import tensorflow as tf
import numpy as np
from scipy import sparse
import json


class LearnModel(object):
    def __int__(self, rule_length, training_iteration, learning_rate, regularization_rate,
                fact_dic, entity_size, candidate, pt, isUncertain):
        self.rule_length = rule_length
        self.training_iteration = training_iteration
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        # modify!
        self.model_name = 'learnModel_trainIter=%d_lr=%f' % (training_iteration, learning_rate)
        self.fact_dic = fact_dic
        self.entity_size = entity_size
        self.rule_num = len(candidate)
        self.train_body = np.array([item[0] for item in candidate])
        self.train_head = pt
        self.para_w = None
        self.isUncertain = isUncertain

    def get_matrix(self, p):
        # sparse matrix
        pfacts = self.fact_dic.get(p)
        pmatrix = sparse.dok_matrix((self.entity_size, self.entity_size), dtype=np.int32)
        if self.isUncertain is True:
            for f in pfacts:
                pmatrix[f[0], f[1]] = f[2]
        else:
            for f in pfacts:
                pmatrix[f[0], f[1]] = 1
        return pmatrix

    def train(self):
        print("\nself.traindata:")
        print(self.train_body)
        loss_history = []
        # batch training: the minimize the overall loss.
        with tf.Graph().as_default(), tf.Session() as sess:
            # define the model parameters
            # rule_index = tf.constant([i for i in range(self.rule_length)])
            # x_body = tf.placeholder(shape=[self.rule_length], dtype=tf.int32)
            # y_head = tf.placeholder(shape=[1], dtype=tf.int32)
            x_body = self.train_body
            y_head = self.train_head
            self.para_w = tf.get_variable('w', shape=[self.rule_num], dtype=tf.float32,
                                          initializer=tf.random_normal_initializer,
                                          regularizer=tf.contrib.layers.l2_regularizer(self.regularization_rate))

            # sparse matrix
            # get matrix operation
            M_R = None
            norm2_loss = 0.0
            M_R_t = self.get_matrix(y_head)
            for i in range(self.rule_num):
                index = -1
                for j in range(self.rule_length):
                    if index == -1:
                        index = 0
                        M_R = self.get_matrix(x_body[i][j])
                    else:
                        M_R = M_R.dot(self.get_matrix(x_body[i][j]))
                M_R = M_R.todok()
                # define loss and train_step
                _loss = 0.0
                for key in M_R.keys():
                    _loss = _loss + np.square(tf.slice(self.para_w, [i], [1]) * M_R[key] - M_R_t[key])
                    M_R_t[key] = -999
                for key in M_R_t.keys():
                    if M_R_t[key] != -999:
                        _loss = _loss + np.square(tf.slice(self.para_w, [i], [1]) * M_R[key] - M_R_t[key])
                norm2_loss = norm2_loss + tf.sqrt(_loss)
            loss = norm2_loss + tf.nn.l2_loss(self.para_w)*self.regularization_rate
            # AdagradOptimizer GradientDescentOptimizer
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            print("Training begins.")
            # initialize all variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for step in range(self.training_iteration):
                # every iteration is for all data
                print("Iteration %d: " % step)
                sess.run(train_step)
                print(' w = ' + str(sess.run(self.para_w)))
                temp_loss = sess.run(loss)
                # print("index: " + str(sess.run(rule_index)))
                # if (i + 1) % 5 == 0:
                print(' loss = ' + str(temp_loss) + "\n")
                loss_history.append(temp_loss)
            saver = tf.train.Saver()
            saver.save(sess, './weights/pt=%s_model:%s.temp' % (str(self.train_head), self.model_name))  # modify!
            self.save_weights(str(sess.run(self.para_w)))
            print("Training ends.")

    def save_weights(self, weights):
        f = open('./weights/pt=%s_model:%s.weights.json' % (str(self.train_head), self.model_name), "w")
        f.write(json.dumps({"weights": weights}))
        f.close()

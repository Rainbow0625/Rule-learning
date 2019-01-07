import tensorflow as tf
import numpy as np
import rule_search_and_learn_weights as r


class LearnModel(object):
    def __int__(self, rule_length, training_iteration, learning_rate, factdic, entitysize):
        self.rule_length = rule_length
        self.training_iteration = training_iteration
        self.learning_rate = learning_rate
        self.model_name = 'learnModel_trainIter=%d_lr=%f.weights' % (training_iteration, learning_rate)
        self.fact_dic = factdic
        self.entity_size = entitysize

    def load_data(self, candidate, pt):
        self.rule_num = len(candidate)
        self.train_body = np.array(candidate)
        self.train_head = pt
        print("\nself.traindata:")
        print(self.train_body)
        '''
        for body_rule in candidate:
            p1 = body_rule[0]
            p2 = body_rule[1]
            r.getmatrix(self.fact_dic, p1, self.entity_size)
        '''

    def loss(self, output, y):
        with tf.variable_scope('Composition', reuse=True):
            weight = tf.get_variable("weight")
        # L2 Regularization + Mean Squared Error
        mse = tf.reduce_sum(tf.square(y - output))
        # mse = tf.losses.mean_squared_error(y, output)
        loss = tf.nn.l2_loss(weight)  + mse
        return loss

    def train(self):
        loss_history = []
        with tf.Graph().as_default(), tf.Session() as sess:
            # define the model structures
            x_body = tf.placeholder(shape=[self.rule_length], dtype=tf.int32)
            y_head = tf.placeholder(shape=[1], dtype=tf.int32)
            w = tf.Variable(tf.random_normal(shape=[1]))
            # get matrix operation!!!!!!!!!!!!!!!
            loss = w
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            # or AdagradOptimizer

            print("Training begins.")
            # initialize all variables
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for step in range(self.training_iteration):
                # every iteration is for all data
                print("  Iteration %d: \n" % step)
                for i in range(len(self.train_body)):
                    print("    Rule body %d: \n" % i)
                    sess.run(train_step, feed_dict={x_body: self.train_body[i], y_head: self.train_head})
                    print('      w = ' + str(sess.run(w)))
                    temp_loss = sess.run(loss, feed_dict={x_body: self.train_body[i], y_head: self.train_head})
                    print('      loss = ' + str(temp_loss))
                    loss_history.append(temp_loss)
                    # if (i + 1) % 5 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './weights/%s.temp' % "pt="+str(self.train_head)+"_modelxxxxx")  # modify!
            print("Training ends.")

    def getWeights(self):
        return 0

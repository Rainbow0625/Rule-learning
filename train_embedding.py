import config.Config as conf
import models
import os


def trainModel(flag, BENCHMARK, work_threads, train_times, nbatches, dimension):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # warnings.filterwarnings("ignore")
    # print("\nThe benchmark is " + BENCHMARK + ".\n")
    con = conf.Config()  # create Class Config()
    if flag == 0:
        file = "before"
        con.set_in_path("./benchmarks/"+BENCHMARK+"/")
    elif flag == 1:
        file = "after"
        con.set_in_path("./sampled/"+BENCHMARK+"/")

    # True: Input test files from the same folder.
    # con.set_test_flag(True)

    con.set_work_threads(work_threads)  # 4
    con.set_train_times(train_times)  # 100
    con.set_nbatches(nbatches)  # 100
    con.set_alpha(0.001)
    con.set_bern(0)
    con.set_margin(1.0)  #what??
    con.set_dimension(dimension)  # 100
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("Adagrad")

    con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files("./embedding/"+file + "/" + BENCHMARK+"/model.vec.tf", 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files("./embedding/"+file + "/" + BENCHMARK+"/embedding.vec.json")  # because of the big data!
    # Initialize experimental settings.
    con.init()
    # Set the knowledge embedding model
    con.set_model(models.TransE)  # DistMult
    # Train the model.
    con.run()
    print("\nTrain successfully!")

    # To test models after training needs "set_test_flag(True)".
    con.test()
    con.show_link_prediction(2, 1)  # what does the para mean?
    con.show_triple_classification(2, 1, 3)

    return con.get_parameters_by_name("ent_embeddings"), con.get_parameters_by_name("rel_embeddings")

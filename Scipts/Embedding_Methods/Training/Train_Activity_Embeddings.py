import gensim

def train_model(log, vector_size = 8, window_size=2, min_count=1, sg=1, epochs=300):
    #some default values that not necesserily make sense
    model = gensim.models.Word2Vec(log, vector_size= vector_size, window=window_size, 
                                   min_count=min_count, sg = sg)
    model.train(log, total_examples=len(log), epochs=epochs)
    return model
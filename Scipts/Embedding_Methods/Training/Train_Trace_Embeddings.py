import gensim

def train_model(taggedlog, vector_size = 8, window_size=2, min_count=1, dm=0, epochs=300):
    #some default values that not necesserily make sense
    model = gensim.models.Doc2Vec(taggedlog, vector_size=vector_size, window=window_size, 
                                  min_count=1, dm = 0)
    model.train(taggedlog, total_examples=len(taggedlog), epochs=300)
    return model


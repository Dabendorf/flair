# Dies ist die Testmethode die das Modell testen soll

from flair.models import SequenceTagger
from flair.data import Sentence
from flair.models import MultiTagger
from flair.models.sandbox.srl_tagger import SemanticRoleTagger 
from flair.trainers import ModelTrainer
from flair.data import Corpus
import flair.datasets

# Load Corpus
corpus = flair.datasets.UP_ENGLISH()

# Tag Type TODO
#tag_type = 'frame'

# Generate Tag Dictionary
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
#print(len(tag_dictionary))

# Init Sequence Tagger
tagger: SequenceTagger = SemanticRoleTagger(rnn_hidden_size=10, # 256
										embeddings=embeddings,
										tag_dictionary=tag_dictionary,
										#tag_type=tag_type,
										#embedding_size = ,
                                        #rnn_type="LSTM",
                                        rnn_layers=2,
										use_crf=False) # true macht Probleme beim Trainieren


# Train model
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train("resources/order/test",
			  learning_rate=0.1,
			  mini_batch_size=32,
			  max_epochs=10,
              train_with_dev=False,
              embeddings_storage_mode="CPU") # {gpu, cpu, none}

result, main_score = tagger.evaluate(corpus.test)
print(result.detailed_results)
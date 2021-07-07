# Dies ist die Testmethode die das Modell testen soll

from flair.models import SequenceTagger
from flair.data import Sentence
from flair.models import MultiTagger
from flair.models.sandbox.simple_sequence_tagger_model import SimpleSequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus
import flair.datasets
from flair.embeddings import WordEmbeddings

# Load Corpus
corpus = flair.datasets.UP_ENGLISH()

glove_embedding = WordEmbeddings('glove')

# Generate Tag Dictionary
# Labels of semantic roles, use add item
tag_type = "frame"
tag_dictionary = corpus.make_label_dictionary(tag_type)

# Init Sequence Tagger
tagger: SequenceTagger = SimpleSequenceTagger(embeddings=glove_embedding,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)


# Train model
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train("resources/order/test",
			  learning_rate=0.1,
			  mini_batch_size=32,
			  max_epochs=10,
              train_with_dev=False,
              embeddings_storage_mode="CPU") #auf cuda ändern

result, main_score = tagger.evaluate(corpus.test)
print(result.detailed_results)
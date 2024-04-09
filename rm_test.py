import dspy.retrieve
from dspy.retrieve.chromadb_rm import ChromadbRM
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.batch_utils import create_batches
import os
import dspy
from colbert.infra.config.config import ColBERTConfig

load_dotenv(override=True)
emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small")

crm = ChromadbRM(
    collection_name="rows",
    persist_directory="/home/athekunal/DSPy-contributions/Text-to-SQL/India_TABLE",
    embedding_function=emb_fn
)
reranker = dspy.ColBERTv2RerankerLocal(checkpoint_name='colbert-ir/colbertv2.0',colbert_config=ColBERTConfig())
dspy.settings.configure(rm=crm,reranker=reranker)

ret_rerank = dspy.RetrieveThenRerank(k=3)

print(ret_rerank(["Software Internet","Packaging and Container"],k=3))
print(ret_rerank(["Software Internet"],k=3))

# retriever = dspy.Retrieve(k=2)

# print(retriever(["Software Internet"],by_prob=False,where={"table_name":"capexIndia"}))
# print("-"*100)
# print(retriever(["Software Internet","Packaging"],by_prob=False,where={"table_name":"capexIndia"}))
# import dspy

# colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
# dspy.settings.configure(rm=colbertv2_wiki17_abstracts,reranker=colbertv2_wiki17_abstracts)

# #Define Retrieve Module
# retriever = dspy.RetrieveThenRerank(k=3)

# query='When was the first FIFA World Cup held?'

# # Call the retriever on a particular query.
# topK_passages = retriever([query])

# for idx, passage in enumerate(topK_passages):
#     print(f'{idx+1}]', passage, '\n')

# import os
# import dspy
# os.environ['COLBERT_LOAD_TORCH_EXTENSION_VERBOSE'] = "True"
# if __name__ == "__main__":
#     passages =  [
#     "The quick brown fox jumps over the lazy dog.",
#     "She sells seashells by the seashore.",
#     "I am the master of my fate, I am the captain of my soul.",
#     "To be or not to be, that is the question.",
#     "All's fair in love and war.",
#     "A journey of a thousand miles begins with a single step.",
#     "Two wrongs don't make a right.",
#     "The pen is mightier than the sword.",
#     "Actions speak louder than words.",
#     "Beauty is in the eye of the beholder.",
#     "Practice makes perfect.",
#     "Where there's a will, there's a way.",
#     "When in Rome, do as the Romans do.",
#     "The early bird catches the worm.",
#     "You can't judge a book by its cover.",
#     "A picture is worth a thousand words.",
#     "Honesty is the best policy.",
#     "Don't count your chickens before they hatch.",
#     "Every cloud has a silver lining.",
#     "If at first you don't succeed, try, try again.",
#     "Look before you leap.",
#     "Rome wasn't built in a day.",
#     "The grass is always greener on the other side.",
#     "Absence makes the heart grow fonder.",
#     "Actions speak louder than words.",
#     "Ask and you shall receive.",
#     "Better late than never.",
#     "Don't bite the hand that feeds you.",
#     "Don't put all your eggs in one basket.",
#     "Easy come, easy go.",
#     "Every dog has its day.",
#     "Good things come to those who wait.",
#     "It's a piece of cake.",
#     "It's raining cats and dogs.",
#     "Kill two birds with one stone.",
#     "Let sleeping dogs lie.",
#     "Like father, like son.",
#     "Make hay while the sun shines.",
#     "Necessity is the mother of invention.",
#     "Out of sight, out of mind.",
#     "Patience is a virtue.",
#     "Practice what you preach.",
#     "The best things in life are free.",
#     "The squeaky wheel gets the grease.",
#     "There's no place like home.",
#     "Too many cooks spoil the broth.",
#     "When the going gets tough, the tough get going.",
#     "You reap what you sow.",
#     "A watched pot never boils.",
#     "Actions speak louder than words.",
#     "An apple a day keeps the doctor away.",
#     "Beggars can't be choosers.",
#     "Curiosity killed the cat.",
#     "Don't cry over spilled milk.",
#     "Don't put off until tomorrow what you can do today.",
#     "Every cloud has a silver lining.",
#     "Fortune favors the bold.",
#     "If the shoe fits, wear it.",
#     "It takes two to tango.",
#     "Keep your friends close and your enemies closer.",
#     "Let bygones be bygones.",
#     "No pain, no gain.",
#     "Once bitten, twice shy.",
#     "Practice makes perfect.",
#     "The apple doesn't fall far from the tree.",
#     "The early bird catches the worm.",
#     "The grass is always greener on the other side.",
#     "The more, the merrier.",
#     "There's no such thing as a free lunch.",
#     "To kill two birds with one stone.",
#     "When in Rome, do as the Romans do.",
#     "You can't have your cake and eat it too.",
#     "You can't make an omelet without breaking eggs.",
#     "A friend in need is a friend indeed.",
#     "A penny saved is a penny earned.",
#     "Actions speak louder than words.",
#     "Beauty is in the eye of the beholder.",
#     "Better late than never.",
#     "Don't count your chickens before they hatch.",
#     "Don't put all your eggs in one basket.",
#     "Every cloud has a silver lining.",
#     "If at first you don't succeed, try, try again.",
#     "If you can't beat them, join them.",
#     "Necessity is the mother of invention.",
#     "One man's trash is another man's treasure.",
#     "Practice makes perfect.",
#     "The early bird catches the worm.",
#     "The grass is always greener on the other side.",
#     "There's no place like home.",
#     "Too many cooks spoil the broth.",
#     "When in Rome, do as the Romans do.",
#     "You can't judge a book by its cover.",
#     "You reap what you sow.",
#     "A bird in the hand is worth two in the bush.",
#     "A penny for your thoughts.",
#     "Actions speak louder than words.",
#     "All good things must come to an end.",
#     "Beauty is only skin deep.",
#     "Don't bite the hand that feeds you.",
#     "Don't put off until tomorrow what you can do today.",
#     "Every dog has its day.",
#     "Fortune favors the bold.",
#     "If you want something done right, do it yourself.",
#     "It's better to be safe than sorry.",
#     "Make hay while the sun shines.",
#     "Necessity is the mother of invention.",
#     "Out of sight, out of mind.",
#     "Practice what you preach.",
#     "The best things in life are free.",
#     "The early bird catches the worm."
# ]

#     col = dspy.ColBERTv2Local(passages=passages)
    
    # searcher = col.get_index(passages=passages[:10])
    # res = searcher.get_docs(searcher,query="Software",k=5)
    # print(res)

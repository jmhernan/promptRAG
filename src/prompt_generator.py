import os
from src.create_vdb import IndexTextEmbeddings
import src.text_preprocess as tp



class TweetPromptGenerator:
    def __init__(self, dataset_with_index, model_name, project_root, template_path=None):
        self.dataset_with_index = dataset_with_index
        self.embeddings_generator = self._initialize_embeddings_generator(model_name)
        self.prompt_template = self._load_prompt_template(project_root, template_path)
    
    def _initialize_embeddings_generator(self, model_name):
        embedding_generator = IndexTextEmbeddings(model_name)
        return embedding_generator
    
    def _load_prompt_template(self, project_root, template_path):
        if template_path is None:
            template_path = os.path.join(project_root, 
                                        'promptRAG/src/prompt_templates/promptTemplate.txt')
        with open(template_path, 'r') as file:
            prompt_template = file.read()
        return prompt_template
    
    def query_faiss_index(self, text, k=5):
        all_results = []
        embeddings = self.embeddings_generator.get_embeddings(text)
        for embedding in embeddings:
            scores, samples = self.dataset_with_index.get_nearest_examples("embeddings", embedding, k=k)
            all_results.append((scores, samples))
        return all_results

    def format_retrieved_texts(self, query_results):
        formatted_texts = ""
        for query_idx, result in enumerate(query_results):
            scores, samples = result
            row_ids = samples['rowid']
            texts = samples['story']
            for score, row_id, text in zip(scores, row_ids, texts):
                # formatted_texts += f"Document ID {row_id} (Similarity Score: {score:.2f}):\n{text}\n\n"
                formatted_texts += f"- {text}\n"
        return formatted_texts, texts, samples['code'], scores

    def generate_prompts_for_tweets(self, tweets, k=3, clean_tweets=False):
        prompts = []
        for tweet in tweets:
            try:
                original_tweet = tweet
                if clean_tweets:
                    tweets = tp.clean_text(tweet)
                query_results = self.query_faiss_index([tweet], k=k)
                formatted_texts, texts, themes, scores = self.format_retrieved_texts(query_results)
                complete_prompt = self.prompt_template.format(tweet=original_tweet, retrieved_texts=formatted_texts)
                prompts.append((complete_prompt, texts, themes, scores))
            except Exception as e:
                print(f"Error processing tweet: '{tweet}'. Error: {e}")
                continue
        return prompts
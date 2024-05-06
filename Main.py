huggingface_token = "hf_AmwHEORkXCzuCSJDxVJVcQNuQRRmO"

from pymed import PubMed
from typing import List
from haystack import component
from haystack import Document
from transformers import AutoModelForCausalLM, AutoTokenizer

import streamlit as st
st.title("MEDITRON....!!!")


pubmed = PubMed(tool="Haystack2.0Prototype", email="tilde.thurium@deepset.ai")
def documentize(article):
  return Document(content=article.abstract, meta={'title': article.title, 'keywords': article.keywords})

@component
class PubMedFetcher():

  @component.output_types(articles=List[Document])
  def run(self, queries: list[str]):
    cleaned_queries = queries[0].strip().split('\n')

    articles = []
    try:
      for query in cleaned_queries:
        response = pubmed.query(query, max_results = 1)
        documents = [documentize(article) for article in response]
        articles.extend(documents)
    except Exception as e:
        print(e)
        print(f"Couldn't fetch articles for queries: {queries}" )
        return {'articles' : "not_found"}
    results = {'articles': articles}
    return results

from haystack.components.generators import HuggingFaceTGIGenerator

keyword_llm = HuggingFaceTGIGenerator("mistralai/Mixtral-8x7B-Instruct-v0.1", token=huggingface_token)
keyword_llm.warm_up()

llm = HuggingFaceTGIGenerator("mistralai/Mixtral-8x7B-Instruct-v0.1", token=huggingface_token)
llm.warm_up()
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder

keyword_prompt_template = """
Your task is to convert the following question into 3 keywords that can be used to find relevant medical research papers on PubMed.
Here is an examples:
question: "What are the latest treatments for major depressive disorder?"
keywords:
Antidepressive Agents
Depressive Disorder, Major
Treatment-Resistant depression
---
question: {{ question }}
keywords:
"""

prompt_template = """
Answer the question truthfully based on the given documents.
If the documents don't contain an answer, use your existing knowledge base.

q: {{ question }}
Articles:
{% for article in articles %}
  {{article.content}}
  keywords: {{article.meta['keywords']}}
  title: {{article.meta['title']}}
{% endfor %}

"""
keyword_prompt_builder = PromptBuilder(template=keyword_prompt_template)

prompt_builder = PromptBuilder(template=prompt_template)
fetcher = PubMedFetcher()

pipe = Pipeline()

pipe.add_component("keyword_prompt_builder", keyword_prompt_builder)
pipe.add_component("keyword_llm", keyword_llm)
pipe.add_component("pubmed_fetcher", fetcher)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("keyword_prompt_builder.prompt", "keyword_llm.prompt")
pipe.connect("keyword_llm.replies", "pubmed_fetcher.queries")

pipe.connect("pubmed_fetcher.articles", "prompt_builder.articles")
pipe.connect("prompt_builder.prompt", "llm.prompt")

def ask(question):
  output = pipe.run(data={"keyword_prompt_builder": {"question": question},
                          "prompt_builder": {"question": question},
                          "llm": {"generation_kwargs": {"max_new_tokens": 500}}})
  st.success("Generated successfully")
  st.write(output['llm']['replies'][0])

def body():
  text = st.text_input("Please enter some text to generate image")
  if (st.checkbox("SUBMIT")):
    answer = ask(text)

if __name__ == '__main__':
    body()




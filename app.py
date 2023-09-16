# Dependencies
import os
from apikey import apikey

# Framework for the communication of various services
import streamlit as st

# Framework to work with LLMs
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
# Accessing the tools
from langchain.utilities import WikipediaAPIWrapper

os.environ['HUGGINGFACEHUB_API_TOKEN'] = apikey

# App framework
st.title('🤗 YouTube Content GPT 🦜🔗')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write a youtube video script based on this title TITLE: {title} while leveraging this '
             'wikipedia reserch:{wikipedia_research} '
)

# Memory
# Setting Topic via prompt
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history', ai_prefix="\nAI Assistant",
                                        human_prefix="User")
# Setting Title via prompt
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history', ai_prefix="\nAI Assistant",
                                        human_prefix="User")


# LLM Instantiation
# Flan-T5 - Alternative of GPT-3
# Flan-T5 is an encoder-decoder transformer model that reframes all NLP tasks into a text-to-text format.
# With appropriate prompting, it can perform zero-shot NLP tasks such as text summarization, common sense reasoning,
# natural language inference, question answering, sentence and sentiment classification, translation,
# and pronoun resolution.
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.9, "max_length": 512})

# LangChain took the topic and Script and passed them to the Template
# First chain: Setting Topic via prompt and keeping its previous instances in memory
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
# Second Chain: Setting Title via prompt and keeping its previous instances in memory
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
# If wikipedia research is not available, then sequential chain is of use to synchronize title and script
# It will take the title and feed it into the script
# sequential_chain = SequentialChain(chains=[title_chain, script_chain],
#                                    input_variables=['topic'], output_variables=['title', 'script'])

wiki = WikipediaAPIWrapper()

# Show information according to the value passed by prompt
if prompt:
    title = title_chain.run(prompt)
    # Notice the connection between title and script
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.subheader("Title of Youtube Video:")
    st.write(title)
    st.subheader("Start of Script for Youtube Video:")
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Start of Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)

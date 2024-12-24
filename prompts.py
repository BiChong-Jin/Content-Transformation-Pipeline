from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain
from langchain.llms import OpenAI

# Initialize your LLM (requires OPENAI_API_KEY in env)
llm = OpenAI(temperature=0.7)

# Prompt for Summarization
summarize_prompt = PromptTemplate(
    input_variables=["input_text"],
    template="""
    You are a helpful assistant. Summarize the following text into a concise paragraph:
    {input_text}
    """
)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt, output_key="summary")

# Prompt for Sentiment Analysis
sentiment_prompt = PromptTemplate(
    input_variables=["input_text"],
    template="""
    You are a sentiment analysis expert. Determine if the text is positive, negative, or neutral and explain briefly:
    {input_text}
    """
)
sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt, output_key="sentiment")

# Prompt for Style Transformation
style_prompt = PromptTemplate(
    input_variables=["summary", "sentiment", "requested_style"],
    template="""
    You are a writing assistant. Given the following summary:
    {summary}

    Sentiment Analysis result:
    {sentiment}

    Rewrite the summary in the following style: {requested_style}
    """
)
style_chain = LLMChain(llm=llm, prompt=style_prompt, output_key="final_text")

# Combine them into a SequentialChain
# We'll run sentiment_chain and summarize_chain in parallel first, then run style_chain.
# For that, we can do a two-step approach or run them sequentially and pass outputs.
# Example: run summarization and sentiment first, then style:
transform_chain = SequentialChain(
    chains=[summarize_chain, sentiment_chain, style_chain],
    input_variables=["input_text", "requested_style"],
    output_variables=["summary", "sentiment", "final_text"],
    verbose=True
)

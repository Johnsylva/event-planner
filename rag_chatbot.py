import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

llm = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
dense_index = pc.Index("llm-project")

assistant_message = "What do you want to do today?"
print(f"Assistant: {assistant_message}\n")
user_input = input("User: ")

history = [
     {"role": "developer", "content": """You are an AI event planner
     who is knowledgeable about various events in Dallas. One such event is a Live Jazz Night in the music category; its a live performance featuring talented local jazz musicians."""},
    {"role": "assistant", "content": assistant_message}
]

while user_input != "exit":
    # RAG Step #1: Retrieve relevant chunks from vector DB
    results = dense_index.search(
        namespace="mock_events",
        query={
            "top_k": 3,
            "inputs": {
                'text': user_input
            }
        }
    )

    # RAG Step #2: Convert chunks into one long string of documentation
    documentation = ""

    for hit in results['result']['hits']:
        fields = hit.get('fields')
        chunk_text = fields.get('chunk_text')
        documentation += chunk_text

    # RAG Step #3: Insert retrieved documentation into prompt
    {"role": "developer", "content": f"""You are an AI event planner
     who is knowledgeable about various events in Dallas. One such event is a Live Jazz Night in the music category; its a live performance featuring talented local jazz musicians
     You are to answer user queries below solely on
     the following documentation: {documentation}"""}

    
    history += [
        {"role": "user",
         "content": f"""Here are excerpts from the mock events file: {documentation}. Use whatever
         info from the above file excerpts (and no other info)
         to answer the following query: {user_input}"""}
    ]

    response = llm.responses.create(
        model="gpt-4.1-mini",
        temperature=0,
        input=history
    )

    print(f"\nAssistant: {response.output_text}\n")

    history += [
        {"role": "assistant", "content": response.output_text},
    ]

    user_input = input("User: ")

print("Goodbye!")
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
llm = OpenAI()

# Load the entire documentation into memory
with open("mock_events.md", "r", encoding="utf-8") as file:
    documentation = file.read()

assistant_message = "what event do you want to do today?"
print(f"Assistant: {assistant_message}\n")
user_input = input("User: ")

history = [
    {"role": "developer", "content": f"""You are an AI event planner
     who is knowledgeable about various events in Dallas. One such event is a Live Jazz Night in the music category; its a live performance featuring talented local jazz musicians
     You are to answer user queries below solely on
     the following documentation: {documentation}"""},
    {"role": "assistant", "content": assistant_message},
    {"role": "user", "content": user_input},
]

while user_input != "exit":
    response = llm.responses.create(
        model="gpt-4.1-mini",
        temperature=0,
        input=history
    )

    print(f"\nAssistant: {response.output_text}")

    user_input = input("\nUser: ")

    history += [
        {"role": "assistant", "content": response.output_text},
        {"role": "user", "content": user_input}
    ]
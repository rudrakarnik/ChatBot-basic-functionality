from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
text = input("Enter the text: ")
response = openai.Completion.create(
  model="text-davinci-003",
  prompt=f"check if the given statement is small talk or not. If it is a small talk Return only 'It's chit-chat' as output else 'It's not a chit-chat' as output\n{text}",
  temperature=0,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(f"{response['choices'][0]['text']}")
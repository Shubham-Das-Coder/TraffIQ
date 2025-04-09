from together import Together

# Replace 'your_api_key' with your actual API key
client = Together(api_key='tgp_v1_SX55lua8c1mUbcpSooZoF6Sinto3hHNm0j7rVjgiII8')

response = client.chat.completions.create(
    model="meta-llama/Llama-Vision-Free",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
)
print(response.choices[0].message.content)

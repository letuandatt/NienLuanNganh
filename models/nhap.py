


with open('data.txt', 'r', encoding='utf-8') as f:
    content = f.read()

print(content)
sentences = content.split('\n')
print(sentences)

for sentence in sentences:
    nhp = sentence.split(', ') # Kiểu list
    print(nhp[0], nhp[1])
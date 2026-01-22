import os
from dotenv import load_dotenv
from pinecone import Pinecone
import re  # regular expression - regex

load_dotenv()


def split_markdown_by_h2(md_text):
    """Converts Markdown file into a list of text chunks.
    Each chunk spans one section of the text, defined by H2s"""
    # regex - pattern is from H2 to next H2 or EOF
    pattern = r"(?m)^## .+?(?=^## |\Z)"
    chunks = re.findall(pattern, md_text, re.DOTALL)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


# Grab the documentation from the mock_events docs
with open("mock_events.md", "r", encoding="utf-8") as file:
    documentation = file.read()

# Pass in the documentation to split it into relevant chunks
chunks = split_markdown_by_h2(documentation)

print(f"Found {len(chunks)} chunks")

# Wrap each chunk in the record format that Pinecone expects:
records = []
for i, chunk in enumerate(chunks):
    records.append({
        "id": f"chunk-{i}",
        "chunk_text": chunk,
        "manual": "mock_events"
    })

# Upload/upsert the chunks into the Pinecone index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
dense_index = pc.Index("llm-project")

# Pinecone can only handle 96 records at a time, so we upload in batches
BATCH_SIZE = 96
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i + BATCH_SIZE]
    dense_index.upsert_records("mock_events", batch)
    print(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} records)")

print(f"Done! Uploaded {len(records)} chunks to Pinecone.")
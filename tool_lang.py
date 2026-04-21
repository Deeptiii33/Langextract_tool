import langextract as lx
import textwrap
import os

#  Setup
PROJECT_ID = "your-project-id"
LOCATION = "your-location"(us-west1 or any)

os.makedirs("test_output", exist_ok=True)

# Prompt
prompt = textwrap.dedent("""\
Extract characters, emotions, and relationships in order of appearance.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.
""")

# Example 
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks?",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
        ]
    )
]

# Input
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# CORRECT extraction (Vertex + langextract)
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-1.5-flash", (use the model which is compatible)

    # THIS IS THE KEY PART(vertex AI)
    language_model_params={
        "vertexai": True,
        "project": PROJECT_ID,
        "location": LOCATION
    }
)

# Debug
print("\n✅ Extracted Entities:")
for e in result.extractions:
    print(f"- {e.extraction_class}: {e.extraction_text}")

if not result.extractions:
    raise ValueError("❌ No extractions found — adjust prompt/input")

# Save (ONLY filename)
lx.io.save_annotated_documents(
    [result],
    output_name="extraction_results.jsonl"
)

print("✅ Saved JSONL to test_output/extraction_results.jsonl")

# Visualization
jsonl_path = "test_output/extraction_results.jsonl"

if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"❌ File not found: {jsonl_path}")

html_content = lx.visualize(jsonl_path)
html_file = "test_output/visualization.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Visualization saved at: {html_file}")

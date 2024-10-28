from pypdf import PdfReader
import json


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path, strict=False)  # Added strict=False to be more lenient
        text_data = []
        
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_data.append(text.strip())
            except Exception as e:
                print(f"Warning: Could not extract text from page: {e}")
                continue
        
        return text_data
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return []

def create_training_data(pdf_path, output_path):
    # Extract text from PDF
    text_data = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(text_data)} pages from the PDF.")
    
    # Create training examples
    training_data = []
    for text in text_data:
        # Split into smaller chunks (e.g., 512 tokens)
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Minimum length check
                training_data.append({
                    "text": chunk,
                    "source": "coastguard_manual"
                })
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    create_training_data("./coastgaurd.pdf", "../data/training_data.json")

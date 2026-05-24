import torch
from PIL import Image, ImageDraw
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


MODEL_NAME = "vidore/colqwen2.5-v0.2"


def make_page(text: str, filename: str) -> Image.Image:
    image = Image.new("RGB", (900, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 120), text, fill="black")
    image.save(filename)
    return image


def main():
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
        print("Warning: CPU will probably be very slow.")

    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

    pages = [
        make_page("This page is about cardiovascular risk and biomarkers.", "page_1.png"),
        make_page("This page is about airline ticket promotions and cashback.", "page_2.png"),
        make_page("This page is about Python functions, modules, and parameters.", "page_3.png"),
    ]

    queries = [
        "Which page talks about cashback promotions?",
        "Which page discusses Python functions?",
        "Which page is related to cardiovascular biomarkers?",
    ]

    batch_images = processor.process_images(pages).to(model.device)
    batch_queries = processor.process_queries(queries).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    for query_idx, query in enumerate(queries):
        best_page_idx = scores[query_idx].argmax().item()
        print()
        print(f"Query: {query}")
        print(f"Best page: page_{best_page_idx + 1}.png")
        print(f"Scores: {scores[query_idx].tolist()}")


if __name__ == "__main__":
    main()
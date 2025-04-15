import requests
import json
import os
import random
import time
import re
import logging
from tqdm import tqdm
from datetime import datetime
import argparse

from search_book_date import get_book_publication_year

### HYPERPARAMETERS ###
# NOTE: ERAS definition is a deliberately contentious choice.
# The idea is to demonstrate how to get an LLM to learn in a way that is culturally relevant to a use case.
# There are overlapping date windows for such historical periods, and many cultures wouldn't use these Euro-centric labels. 
# The idea shows how GRPO-like methods can align an LLM to a "sovereign AI" approach,
# where cultural preferences can be chosen at national-, local-, or business-level
# instead of by a single maximally general LLM provider.
ERAS = {
    "renaissance": (1500, 1650),
    "enlightenment": (1650, 1800),
    "romantic": (1800, 1837),
    "victorian": (1837, 1901),
    "edwardian": (1901, 1920),
    "modern": (1920, 1960)
}
LANGUAGE = "en"
# NOTE: Project Gutenberg is free for any usage under United States law.
# These books are all already public domain. 
# IMPORTANT: If you are based in another country, it is YOUR responsibility to understand what books you can(not) distribute legally.

### CONSTANTS ###
BASE_DIR = "gutenberg_dataset"
LOG_DIR = "logs"

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{BASE_DIR}/{LOG_DIR}/download_log_{timestamp}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler() 
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/{LOG_DIR}", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/full", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/train", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/validation", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/test", exist_ok=True)
    for era in ERAS:
        os.makedirs(f"{BASE_DIR}/{era}", exist_ok=True)

def download_book_content(formats):
    for fmt in ["text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain", "text/html"]:
        if fmt in formats:
            try:
                response = requests.get(formats[fmt])
                if response.status_code == 200:
                    return response.text
            except:
                continue
    return None

def load_book_cache():
    cache_path = f"{BASE_DIR}/full/book_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_book_cache(cache):
    cache_path = f"{BASE_DIR}/full/book_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_books_for_all_eras(target_books_per_era, logger):
    """Collect books for all eras with better balancing."""
    books_by_era = {era: [] for era in ERAS}
    all_books_seen = set()
    book_cache = load_book_cache()
    page = 1
    
    # Max books to collect per era (add some buffer)
    max_per_era = {era: target_books_per_era * 1.2 for era in ERAS}
    
    # First pass: Use cached books we already know about
    for book_id, book_info in book_cache.items():
        est_year = book_info.get("year")
        if est_year:
            est_year = int(est_year)
            for era, (start_year, end_year) in ERAS.items():
                if start_year <= est_year <= end_year:
                    if len(books_by_era[era]) < max_per_era[era]:
                        # Need to get full book info from Gutendex
                        # Add to a "to fetch" list for second pass
                        break
    
    needs_more_books = lambda: any(len(books_by_era[era]) < target_books_per_era for era in ERAS)
    
    with tqdm(total=sum(max(0, target_books_per_era - len(books_by_era[era])) for era in ERAS), 
              desc="Gathering books for all eras") as pbar:
        
        while needs_more_books() and page < 1000:  # Set a reasonable page limit
            # Prioritize eras that need more books
            deficit_eras = [era for era in ERAS if len(books_by_era[era]) < target_books_per_era]
            
            if not deficit_eras:
                break
                
            res = requests.get(f"https://gutendex.com/books/?page={page}").json()
            if not res.get("results"):
                break
                
            for book in res["results"]:
                # Skip if already processed or doesn't meet criteria
                if book["id"] in all_books_seen or LANGUAGE not in book["languages"]:
                    continue
                    
                all_books_seen.add(book["id"])
                
                # Look for publication year
                est_year = None
                if str(book["id"]) in book_cache:
                    est_year = book_cache[str(book["id"])].get("year")
                    if est_year:
                        est_year = int(est_year)
                else:
                    est_year = get_book_publication_year(book['title'])
                    book_cache[str(book["id"])] = {"title": book["title"], "year": est_year}
                    
                    if len(book_cache) % 10 == 0:
                        save_book_cache(book_cache)
                
                if est_year is None:
                    continue
                
                # Place in appropriate era, prioritizing deficit eras
                for era in deficit_eras:
                    start_year, end_year = ERAS[era]
                    if start_year <= est_year <= end_year:
                        if len(books_by_era[era]) < target_books_per_era:
                            book["estimated_date"] = est_year
                            books_by_era[era].append(book)
                            pbar.update(1)
                            logger.info(f'PLACED - book: `{book["title"]}` id:{book["id"]} in era {era} ({start_year}-{end_year}), year {est_year}.')
                            break
            
            page += 1
            time.sleep(0.1)
    
    save_book_cache(book_cache)
    return books_by_era

def extract_passages(text, min_length=200, max_length=400, num_passages=10):
    """Extract `num_passages` coherent passages from a Project Gutenberg text."""

    # Remove metadata headers/footers in all Project Gutenberg files.
    start_re = r"\*\*\* START OF (.*?) \*\*\*"
    end_re = r"\*\*\* END OF (.*?) \*\*\*"
    start_match = re.search(start_re, text)
    end_match = re.search(end_re, text)
    if start_match:
        text = text[start_match.end():]
    if end_match:
        text = text[:end_match.start()]

    # Normalize whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Split into paragraphs by double newlines (standard Gutenberg formatting)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Filter paragraphs by word length
    valid_paragraphs = [
        p for p in paragraphs
        if min_length <= len(p.split()) <= max_length
    ]

    # If not enough, truncate longer paragraphs to fit
    if len(valid_paragraphs) < num_passages:
        longer_paragraphs = [p for p in paragraphs if len(p.split()) > max_length]
        for lp in longer_paragraphs:
            truncated = ' '.join(lp.split()[:max_length])
            valid_paragraphs.append(truncated)
            if len(valid_paragraphs) >= num_passages:
                break

    # Return a random selection, or all, if it is a smallish book.
    return random.sample(valid_paragraphs, min(num_passages, len(valid_paragraphs)))

def save_passage(passage_data, era, book_id, idx):
    path = f"{BASE_DIR}/{era}/{book_id}_passage_{idx}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(passage_data, f, indent=2)

def process_era_books(era, books, passages_per_book, logger, batch_size=25):
    """Process books for an era with checkpointing and batching."""
    logger.info(f"Processing {len(books)} books for {era}...")
    
    # Load checkpoint if exists
    processed_ids = set(load_checkpoint(era, logger))
    books_to_process = [b for b in books if b["id"] not in processed_ids]
    processed_books = [b for b in books if b["id"] in processed_ids]
    
    if len(processed_ids) > 0:
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} books already processed, {len(books_to_process)} remaining")
    
    # Load existing passages, if any.
    era_passages = []
    era_passages_path = os.path.join(BASE_DIR, era, "passages.json")
    if os.path.exists(era_passages_path):
        try:
            with open(era_passages_path, "r", encoding="utf-8") as f:
                era_passages = json.load(f)
                logger.info(f"Loaded {len(era_passages)} existing passages for {era}")
        except Exception as e:
            logger.error(f"Error loading passages for {era}: {e}")
    
    # Process batches.
    total_batches = (len(books_to_process) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(books_to_process))
        batch_books = books_to_process[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} for {era} ({len(batch_books)} books)")
        batch_passages = []
        batch_processed = []
        
        for book in tqdm(batch_books, desc=f"Processing batch {batch_idx+1} for {era}"):
            try:
                content = download_book_content(book["formats"])
                if not content:
                    logger.warning(f"Could not download content for book: {book['title']} id:{book['id']}")
                    continue
                    
                passages = extract_passages(content, num_passages=passages_per_book)
                logger.info(f"Extracted {len(passages)} passages from book: {book['title']} id:{book['id']}")
                
                for idx, passage in enumerate(passages):
                    date = book["estimated_date"]
                    pdata = {
                        "passage": passage,
                        "book_id": book["id"],
                        "title": book["title"],
                        "author": book["authors"][0]["name"] if book["authors"] else "Unknown",
                        "era": era,
                        "date": str(date),
                        "clues": [],
                        "rationale": ""
                    }
                    save_passage(pdata, era, book["id"], idx)
                    batch_passages.append(pdata)
                
                batch_processed.append(book)
                processed_books.append(book)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error processing book {book['id']} ({book['title']}): {e}")
        
        save_checkpoint(era, processed_books, logger)
        era_passages.extend(batch_passages)
        batch_file = os.path.join(BASE_DIR, era, f"passages_batch_{batch_idx+1}.json")
        save_json(batch_passages, batch_file)
        save_json(era_passages, era_passages_path)
        logger.info(f"Completed batch {batch_idx+1}/{total_batches} for {era}: {len(batch_passages)} passages collected")
    
    return era_passages

def create_train_val_split(passages, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(passages)
    split_idx = int(len(passages) * (1 - val_ratio))
    return passages[:split_idx], passages[split_idx:]

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_checkpoint(era, processed_books, logger):
    checkpoint_path = f"{BASE_DIR}/checkpoint_{era}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump([b["id"] for b in processed_books], f)
    logger.info(f"Saved checkpoint for {era} with {len(processed_books)} processed books")

def load_checkpoint(era, logger):
    checkpoint_path = f"{BASE_DIR}/checkpoint_{era}.json"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            processed_ids = json.load(f)
        logger.info(f"Loaded checkpoint for {era}: {len(processed_ids)} books already processed")
        return processed_ids
    return []

def process_era_with_checkpoints(era, books, passages_per_book, logger):
    processed_ids = load_checkpoint(era, logger)
    books_to_process = [b for b in books if b["id"] not in processed_ids]
    logger.info(f"Processing {len(books_to_process)} remaining books for {era}")
    
    # Add loading existing passages
    existing_passages = []
    passage_path = f"{BASE_DIR}/{era}/passages.json"
    if os.path.exists(passage_path):
        with open(passage_path, 'r') as f:
            existing_passages = json.load(f)
    
    new_passages = process_era_books(era, books_to_process, passages_per_book, logger)
    all_passages = existing_passages + new_passages
    return all_passages

def main(books_per_era=100, passages_per_book=250, batch_size=25, eras=None, collect_only=False, process_only=False):
    create_directories()
    logger = setup_logging()
    logger.info(f"Starting book collection with target of {books_per_era} books per era")
    
    active_eras = [era for era in (eras or ERAS.keys()) if era in ERAS]
    if not active_eras:
        logger.error("No valid eras specified")
        return

    books_by_era = {}
    if not process_only:
        logger.info(f"Collecting books for eras: {', '.join(active_eras)}")
        books_by_era = get_books_for_all_eras(books_per_era, logger) # The expensive step, calls Perplexity search.
        
        for era, books in books_by_era.items():
            books_file = os.path.join(BASE_DIR, era, "books.json")
            save_json(books, books_file)
    else:
        for era in active_eras:
            books_file = os.path.join(BASE_DIR, era, "books.json")
            if os.path.exists(books_file):
                with open(books_file, "r", encoding="utf-8") as f:
                    books_by_era[era] = json.load(f)
                logger.info(f"Loaded {len(books_by_era[era])} books for {era}")
            else:
                logger.warning(f"No books file found for {era} in process-only mode")
    
    if collect_only:
        logger.info("Collect-only mode, skipping passage extraction")
        return
    
    all_passages = []
    for era, books in books_by_era.items():
        if books:
            era_passages = process_era_books(era, books, passages_per_book, logger, batch_size)
            all_passages.extend(era_passages)
            logger.info(f"Completed processing for {era}: {len(era_passages)} passages collected")
    
    if all_passages:
        save_json(all_passages, os.path.join(BASE_DIR, "full", "passages.json"))
        train, val = create_train_val_split(all_passages)
        val, test = create_train_val_split(val, val_ratio=0.5)
        save_json(train, os.path.join(BASE_DIR, "train", "passages.json"))
        save_json(val, os.path.join(BASE_DIR, "validation", "passages.json"))
        save_json(test, os.path.join(BASE_DIR, "test", "passages.json"))

    logger.info(f"=== Dataset Creation Complete ===")
    logger.info(f"Total passages: {len(all_passages)}")
    if all_passages:
        logger.info(f"Train set: {len(train)} passages")
        logger.info(f"Validation set: {len(val)} passages")
        logger.info(f"Test set: {len(test)} passages")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nb", "--books-per-era", type=int, default=100, help="Minimum number of books per era.")
    parser.add_argument("-np", "--passages-per-book", type=int, default=250, help="Minimum number of passages per book.")
    parser.add_argument("-bs", "--batch-size", type=int, default=25, help="Number of books to process in each batch.")
    parser.add_argument("-e", "--eras", nargs="+", help="Specific eras to process (default: all)")
    parser.add_argument("--collect-only", action="store_true", help="Only collect book metadata, don't extract passages")
    parser.add_argument("--process-only", action="store_true", help="Skip collection, only process already collected books")
    args = parser.parse_args()

    main(
        books_per_era=args.books_per_era, 
        passages_per_book=args.passages_per_book, 
        batch_size=args.batch_size,
        eras=args.eras,
        collect_only=args.collect_only,
        process_only=args.process_only
    )
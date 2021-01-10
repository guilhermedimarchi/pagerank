import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    qty_pages = len(corpus)
    qty_links = len(corpus[page])
    if qty_links == 0:
        return initialize_model_with_value(corpus, 1 / qty_pages)

    model = initialize_model_with_value(corpus, (1 - damping_factor) * (1 / qty_pages))
    for next_page in corpus[page]:
        model[next_page] += damping_factor * (1 / qty_links)
    return model


def initialize_model_with_value(corpus, value):
    model = {}
    for key in corpus:
        model[key] = value
    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_counter = initialize_model_with_value(corpus, 0)
    page = random.choice(list(corpus.keys()))
    for i in range(n):
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), model.values())[0]
        page_counter[page] += 1 / n
    return page_counter


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    for key in corpus.keys():
        if len(corpus[key]) == 0:
            corpus[key] = list(corpus.keys())

    total_pages = len(corpus.keys())
    model = initialize_model_with_value(corpus, 1 / total_pages)
    can_improve = True
    while can_improve:
        can_improve = False
        for key in model.keys():
            old = model[key]
            new = pr(corpus, key, damping_factor, model)
            model[key] = new
            if abs(old - new) > 0.001:
                can_improve = True
    return model


def pr(corpus, page, damping_factor, model):
    sum = 0
    for parent in parents(corpus, page):
        sum += model[parent] / num_links(corpus, parent)
    return ((1-damping_factor) / len(corpus.keys())) + (damping_factor * sum)


def num_links(corpus, page):
    return len(corpus[page])


def parents(corpus, page):
    p = []
    for key in corpus.keys():
        if page in corpus[key]:
            p.append(key)
    return p


if __name__ == "__main__":
    main()

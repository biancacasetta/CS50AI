import os
import random
import re
import sys
import numpy as np

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
    distribution = {page: (1 - damping_factor) / len(corpus) for page in corpus}

    if corpus[page]:
        for link in corpus[page]:
            distribution[link] += damping_factor / len(corpus[page])
    else:  
        distribution = {page: 1 / len(corpus) for page in corpus}
            
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    ranks = {page: 0 for page in corpus}
    random_page = random.choice(list(corpus.keys()))

    for i in range(n):
        tm = transition_model(corpus, random_page, damping_factor)
        random_page = np.random.choice(list(tm.keys()), p=list(tm.values()))
        ranks[random_page] += 0.0001

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {page: 1 / len(corpus) for page in corpus}
    keep_going = True

    while keep_going:
        keep_going = False
        new_ranks = dict()

        for p in corpus:
            linked_probability = 0
            for i in corpus:
                if p in corpus[i]:
                    linked_probability += ranks[i] / len(corpus[i])
                elif not corpus[i]:
                    linked_probability += ranks[i] / len(corpus)
            new_ranks[p] = (1 - damping_factor) / len(corpus) + damping_factor * linked_probability
        
        for key in ranks:
            if abs(ranks[key] - new_ranks[key]) > 0.001:
                keep_going = True
        
        ranks = new_ranks
    
    return ranks


if __name__ == "__main__":
    main()

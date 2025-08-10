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
    output = {}

    # Assign all pages to output dictionary
    for page_i, pages in corpus.items():
        if page_i not in output:
            output[page_i] = 0.00
        for link in pages:
            if link not in output:
                output[link] = 0.00

    # Using probability damping factor
    if len(corpus[page]):
        average = damping_factor / float(len(corpus[page]))
        for link in corpus[page]:
            output[link] += average
    else:
        average = damping_factor / float(len(output))
        for link in output:
            output[link] += average

    # Using 1- damping factor
    average = (1.00 - damping_factor) / float(len(output))
    for link in output:
        output[link] += average
    return output


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    samples = []
    output = {}

    # Generate the first sample
    r = random.randint(0, len(corpus)-1)
    page = list(corpus)[r]
    samples.append(page)

    prob_distri = transition_model(corpus, page, damping_factor)
    for i in range(n-1):
        page = random.choices(list(prob_distri), list(prob_distri.values()))[0]
        samples.append(page)
        prob_distri = transition_model(corpus, page, damping_factor)

    # Organize output
    for name in corpus:
        output[name] = float(samples.count(name)) / float(n)

    return output


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = {}
    new_rank = {}
    # Assigning ranks
    for link in corpus:
        ranks[link] = 1.0000 / float(len(corpus))

    # Reverse corpus
    reverse = {}
    for page in corpus:
        reverse[page] = set()
    for key, li in corpus.items():
        for i in li:
            if key not in reverse[i]:
                reverse[i].add(key)

    max_val = 1.000
    while (max_val > 0.001):
        new_rank = {}
        for page in ranks:
            new_rank[page] = pagerank_formula(corpus, page, damping_factor, ranks, reverse)

        # Normalise values
        sum_rank = sum(new_rank.values())
        dif = float(1.0000 / sum_rank)

        for link in new_rank:
            new_rank[link] *= dif

        # Keep track of max difference
        max_val = 0.0000
        for page in ranks:
            max_val = max(max_val, abs(new_rank[page] - ranks[page]))

        ranks = new_rank.copy()

    return new_rank


def pagerank_formula(corpus, page, d, ranks, reverse):
    # 1 - d
    rank = (1.0000 - d) / float(len(corpus))

    # d
    ind_rank = 0

    for link in reverse[page]:
        ind_rank += ranks[link] / float(len(corpus[link]))

    # Fomula
    rank += d * ind_rank

    return rank
    

if __name__ == "__main__":
    main()
